use std::ffi::OsStr;
use std::fmt::Write;
use std::fs::{self, File};
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::lazy::SyncLazy as Lazy;
use std::path::{Component, Path, PathBuf};

use itertools::Itertools;
use rustc_data_structures::flock;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use serde::Serialize;

use super::{collect_paths_for_type, ensure_trailing_slash, Context, BASIC_KEYWORDS};
use crate::clean::Crate;
use crate::config::{EmitType, RenderOptions};
use crate::docfs::PathError;
use crate::error::Error;
use crate::html::{layout, static_files};
use crate::{try_err, try_none};

static FILES_UNVERSIONED: Lazy<FxHashMap<&str, &[u8]>> = Lazy::new(|| {
    map! {
        "FiraSans-Regular.woff2" => static_files::fira_sans::REGULAR2,
        "FiraSans-Medium.woff2" => static_files::fira_sans::MEDIUM2,
        "FiraSans-Regular.woff" => static_files::fira_sans::REGULAR,
        "FiraSans-Medium.woff" => static_files::fira_sans::MEDIUM,
        "FiraSans-LICENSE.txt" => static_files::fira_sans::LICENSE,
        "SourceSerif4-Regular.ttf.woff2" => static_files::source_serif_4::REGULAR2,
        "SourceSerif4-Bold.ttf.woff2" => static_files::source_serif_4::BOLD2,
        "SourceSerif4-It.ttf.woff2" => static_files::source_serif_4::ITALIC2,
        "SourceSerif4-Regular.ttf.woff" => static_files::source_serif_4::REGULAR,
        "SourceSerif4-Bold.ttf.woff" => static_files::source_serif_4::BOLD,
        "SourceSerif4-It.ttf.woff" => static_files::source_serif_4::ITALIC,
        "SourceSerif4-LICENSE.md" => static_files::source_serif_4::LICENSE,
        "SourceCodePro-Regular.ttf.woff2" => static_files::source_code_pro::REGULAR2,
        "SourceCodePro-Semibold.ttf.woff2" => static_files::source_code_pro::SEMIBOLD2,
        "SourceCodePro-It.ttf.woff2" => static_files::source_code_pro::ITALIC2,
        "SourceCodePro-Regular.ttf.woff" => static_files::source_code_pro::REGULAR,
        "SourceCodePro-Semibold.ttf.woff" => static_files::source_code_pro::SEMIBOLD,
        "SourceCodePro-It.ttf.woff" => static_files::source_code_pro::ITALIC,
        "SourceCodePro-LICENSE.txt" => static_files::source_code_pro::LICENSE,
        "NanumBarunGothic.ttf.woff2" => static_files::nanum_barun_gothic::REGULAR2,
        "NanumBarunGothic.ttf.woff" => static_files::nanum_barun_gothic::REGULAR,
        "NanumBarunGothic-LICENSE.txt" => static_files::nanum_barun_gothic::LICENSE,
        "LICENSE-MIT.txt" => static_files::LICENSE_MIT,
        "LICENSE-APACHE.txt" => static_files::LICENSE_APACHE,
        "COPYRIGHT.txt" => static_files::COPYRIGHT,
    }
});

enum SharedResource<'a> {
    /// This file will never change, no matter what toolchain is used to build it.
    ///
    /// It does not have a resource suffix.
    Unversioned { name: &'static str },
    /// This file may change depending on the toolchain.
    ///
    /// It has a resource suffix.
    ToolchainSpecific { basename: &'static str },
    /// This file may change for any crate within a build, or based on the CLI arguments.
    ///
    /// This differs from normal invocation-specific files because it has a resource suffix.
    InvocationSpecific { basename: &'a str },
}

impl SharedResource<'_> {
    fn extension(&self) -> Option<&OsStr> {
        use SharedResource::*;
        match self {
            Unversioned { name }
            | ToolchainSpecific { basename: name }
            | InvocationSpecific { basename: name } => Path::new(name).extension(),
        }
    }

    fn path(&self, cx: &Context<'_>) -> PathBuf {
        match self {
            SharedResource::Unversioned { name } => cx.dst.join(name),
            SharedResource::ToolchainSpecific { basename } => cx.suffix_path(basename),
            SharedResource::InvocationSpecific { basename } => cx.suffix_path(basename),
        }
    }

    fn should_emit(&self, emit: &[EmitType]) -> bool {
        if emit.is_empty() {
            return true;
        }
        let kind = match self {
            SharedResource::Unversioned { .. } => EmitType::Unversioned,
            SharedResource::ToolchainSpecific { .. } => EmitType::Toolchain,
            SharedResource::InvocationSpecific { .. } => EmitType::InvocationSpecific,
        };
        emit.contains(&kind)
    }
}

impl Context<'_> {
    fn suffix_path(&self, filename: &str) -> PathBuf {
        // We use splitn vs Path::extension here because we might get a filename
        // like `style.min.css` and we want to process that into
        // `style-suffix.min.css`.  Path::extension would just return `css`
        // which would result in `style.min-suffix.css` which isn't what we
        // want.
        let (base, ext) = filename.split_once('.').unwrap();
        let filename = format!("{}{}.{}", base, self.shared.resource_suffix, ext);
        self.dst.join(&filename)
    }

    fn write_shared(
        &self,
        resource: SharedResource<'_>,
        contents: impl 'static + Send + AsRef<[u8]>,
        emit: &[EmitType],
    ) -> Result<(), Error> {
        if resource.should_emit(emit) {
            self.shared.fs.write(resource.path(self), contents)
        } else {
            Ok(())
        }
    }

    fn write_minify(
        &self,
        resource: SharedResource<'_>,
        contents: impl 'static + Send + AsRef<str> + AsRef<[u8]>,
        minify: bool,
        emit: &[EmitType],
    ) -> Result<(), Error> {
        if minify {
            let contents = contents.as_ref();
            let contents = if resource.extension() == Some(OsStr::new("css")) {
                minifier::css::minify(contents).map_err(|e| {
                    Error::new(format!("failed to minify CSS file: {}", e), resource.path(self))
                })?
            } else {
                minifier::js::minify(contents)
            };
            self.write_shared(resource, contents, emit)
        } else {
            self.write_shared(resource, contents, emit)
        }
    }
}

pub(super) fn write_shared(
    cx: &Context<'_>,
    krate: &Crate,
    search_index: String,
    options: &RenderOptions,
) -> Result<(), Error> {
    // Write out the shared files. Note that these are shared among all rustdoc
    // docs placed in the output directory, so this needs to be a synchronized
    // operation with respect to all other rustdocs running around.
    let lock_file = cx.dst.join(".lock");
    let _lock = try_err!(flock::Lock::new(&lock_file, true, true, true), &lock_file);

    // Minified resources are usually toolchain resources. If they're not, they should use `cx.write_minify` directly.
    fn write_minify(
        basename: &'static str,
        contents: impl 'static + Send + AsRef<str> + AsRef<[u8]>,
        cx: &Context<'_>,
        options: &RenderOptions,
    ) -> Result<(), Error> {
        cx.write_minify(
            SharedResource::ToolchainSpecific { basename },
            contents,
            options.enable_minification,
            &options.emit,
        )
    }

    // Toolchain resources should never be dynamic.
    let write_toolchain = |p: &'static _, c: &'static _| {
        cx.write_shared(SharedResource::ToolchainSpecific { basename: p }, c, &options.emit)
    };

    // Crate resources should always be dynamic.
    let write_crate = |p: &_, make_content: &dyn Fn() -> Result<Vec<u8>, Error>| {
        let content = make_content()?;
        cx.write_shared(SharedResource::InvocationSpecific { basename: p }, content, &options.emit)
    };

    fn add_background_image_to_css(
        cx: &Context<'_>,
        css: &mut String,
        rule: &str,
        file: &'static str,
    ) {
        css.push_str(&format!(
            "{} {{ background-image: url({}); }}",
            rule,
            SharedResource::ToolchainSpecific { basename: file }
                .path(cx)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
        ))
    }

    // Add all the static files. These may already exist, but we just
    // overwrite them anyway to make sure that they're fresh and up-to-date.
    let mut rustdoc_css = static_files::RUSTDOC_CSS.to_owned();
    add_background_image_to_css(
        cx,
        &mut rustdoc_css,
        "details.undocumented[open] > summary::before, \
         details.rustdoc-toggle[open] > summary::before, \
         details.rustdoc-toggle[open] > summary.hideme::before",
        "toggle-minus.svg",
    );
    add_background_image_to_css(
        cx,
        &mut rustdoc_css,
        "details.undocumented > summary::before, details.rustdoc-toggle > summary::before",
        "toggle-plus.svg",
    );
    write_minify("rustdoc.css", rustdoc_css, cx, options)?;

    // Add all the static files. These may already exist, but we just
    // overwrite them anyway to make sure that they're fresh and up-to-date.
    write_minify("settings.css", static_files::SETTINGS_CSS, cx, options)?;
    write_minify("noscript.css", static_files::NOSCRIPT_CSS, cx, options)?;

    // To avoid "light.css" to be overwritten, we'll first run over the received themes and only
    // then we'll run over the "official" styles.
    let mut themes: FxHashSet<String> = FxHashSet::default();

    for entry in &cx.shared.style_files {
        let theme = try_none!(try_none!(entry.path.file_stem(), &entry.path).to_str(), &entry.path);
        let extension =
            try_none!(try_none!(entry.path.extension(), &entry.path).to_str(), &entry.path);

        // Handle the official themes
        match theme {
            "light" => write_minify("light.css", static_files::themes::LIGHT, cx, options)?,
            "dark" => write_minify("dark.css", static_files::themes::DARK, cx, options)?,
            "ayu" => write_minify("ayu.css", static_files::themes::AYU, cx, options)?,
            _ => {
                // Handle added third-party themes
                let filename = format!("{}.{}", theme, extension);
                write_crate(&filename, &|| Ok(try_err!(fs::read(&entry.path), &entry.path)))?;
            }
        };

        themes.insert(theme.to_owned());
    }

    if (*cx.shared).layout.logo.is_empty() {
        write_toolchain("rust-logo.png", static_files::RUST_LOGO)?;
    }
    if (*cx.shared).layout.favicon.is_empty() {
        write_toolchain("favicon.svg", static_files::RUST_FAVICON_SVG)?;
        write_toolchain("favicon-16x16.png", static_files::RUST_FAVICON_PNG_16)?;
        write_toolchain("favicon-32x32.png", static_files::RUST_FAVICON_PNG_32)?;
    }
    write_toolchain("brush.svg", static_files::BRUSH_SVG)?;
    write_toolchain("wheel.svg", static_files::WHEEL_SVG)?;
    write_toolchain("clipboard.svg", static_files::CLIPBOARD_SVG)?;
    write_toolchain("down-arrow.svg", static_files::DOWN_ARROW_SVG)?;
    write_toolchain("toggle-minus.svg", static_files::TOGGLE_MINUS_PNG)?;
    write_toolchain("toggle-plus.svg", static_files::TOGGLE_PLUS_PNG)?;

    let mut themes: Vec<&String> = themes.iter().collect();
    themes.sort();

    // FIXME: this should probably not be a toolchain file since it depends on `--theme`.
    // But it seems a shame to copy it over and over when it's almost always the same.
    // Maybe we can change the representation to move this out of main.js?
    write_minify(
        "main.js",
        static_files::MAIN_JS
            .replace(
                "/* INSERT THEMES HERE */",
                &format!(" = {}", serde_json::to_string(&themes).unwrap()),
            )
            .replace(
                "/* INSERT RUSTDOC_VERSION HERE */",
                &format!(
                    "rustdoc {}",
                    rustc_interface::util::version_str().unwrap_or("unknown version")
                ),
            ),
        cx,
        options,
    )?;
    write_minify("search.js", static_files::SEARCH_JS, cx, options)?;
    write_minify("settings.js", static_files::SETTINGS_JS, cx, options)?;

    if cx.include_sources {
        write_minify("source-script.js", static_files::sidebar::SOURCE_SCRIPT, cx, options)?;
    }

    {
        write_minify(
            "storage.js",
            format!(
                "var resourcesSuffix = \"{}\";{}",
                cx.shared.resource_suffix,
                static_files::STORAGE_JS
            ),
            cx,
            options,
        )?;
    }

    if cx.shared.layout.scrape_examples_extension {
        cx.write_minify(
            SharedResource::InvocationSpecific { basename: "scrape-examples.js" },
            static_files::SCRAPE_EXAMPLES_JS,
            options.enable_minification,
            &options.emit,
        )?;
    }

    if let Some(ref css) = cx.shared.layout.css_file_extension {
        let buffer = try_err!(fs::read_to_string(css), css);
        // This varies based on the invocation, so it can't go through the write_minify wrapper.
        cx.write_minify(
            SharedResource::InvocationSpecific { basename: "theme.css" },
            buffer,
            options.enable_minification,
            &options.emit,
        )?;
    }
    write_minify("normalize.css", static_files::NORMALIZE_CSS, cx, options)?;
    for (name, contents) in &*FILES_UNVERSIONED {
        cx.write_shared(SharedResource::Unversioned { name }, contents, &options.emit)?;
    }

    fn collect(path: &Path, krate: &str, key: &str) -> io::Result<(Vec<String>, Vec<String>)> {
        let mut ret = Vec::new();
        let mut krates = Vec::new();

        if path.exists() {
            let prefix = format!(r#"{}["{}"]"#, key, krate);
            for line in BufReader::new(File::open(path)?).lines() {
                let line = line?;
                if !line.starts_with(key) {
                    continue;
                }
                if line.starts_with(&prefix) {
                    continue;
                }
                ret.push(line.to_string());
                krates.push(
                    line[key.len() + 2..]
                        .split('"')
                        .next()
                        .map(|s| s.to_owned())
                        .unwrap_or_else(String::new),
                );
            }
        }
        Ok((ret, krates))
    }

    fn collect_json(path: &Path, krate: &str) -> io::Result<(Vec<String>, Vec<String>)> {
        let mut ret = Vec::new();
        let mut krates = Vec::new();

        if path.exists() {
            let prefix = format!("\"{}\"", krate);
            for line in BufReader::new(File::open(path)?).lines() {
                let line = line?;
                if !line.starts_with('"') {
                    continue;
                }
                if line.starts_with(&prefix) {
                    continue;
                }
                if line.ends_with(",\\") {
                    ret.push(line[..line.len() - 2].to_string());
                } else {
                    // Ends with "\\" (it's the case for the last added crate line)
                    ret.push(line[..line.len() - 1].to_string());
                }
                krates.push(
                    line.split('"')
                        .find(|s| !s.is_empty())
                        .map(|s| s.to_owned())
                        .unwrap_or_else(String::new),
                );
            }
        }
        Ok((ret, krates))
    }

    use std::ffi::OsString;

    #[derive(Debug)]
    struct Hierarchy {
        elem: OsString,
        children: FxHashMap<OsString, Hierarchy>,
        elems: FxHashSet<OsString>,
    }

    impl Hierarchy {
        fn new(elem: OsString) -> Hierarchy {
            Hierarchy { elem, children: FxHashMap::default(), elems: FxHashSet::default() }
        }

        fn to_json_string(&self) -> String {
            let mut subs: Vec<&Hierarchy> = self.children.values().collect();
            subs.sort_unstable_by(|a, b| a.elem.cmp(&b.elem));
            let mut files = self
                .elems
                .iter()
                .map(|s| format!("\"{}\"", s.to_str().expect("invalid osstring conversion")))
                .collect::<Vec<_>>();
            files.sort_unstable();
            let subs = subs.iter().map(|s| s.to_json_string()).collect::<Vec<_>>().join(",");
            let dirs =
                if subs.is_empty() { String::new() } else { format!(",\"dirs\":[{}]", subs) };
            let files = files.join(",");
            let files =
                if files.is_empty() { String::new() } else { format!(",\"files\":[{}]", files) };
            format!(
                "{{\"name\":\"{name}\"{dirs}{files}}}",
                name = self.elem.to_str().expect("invalid osstring conversion"),
                dirs = dirs,
                files = files
            )
        }
    }

    if cx.include_sources {
        let mut hierarchy = Hierarchy::new(OsString::new());
        for source in cx
            .shared
            .local_sources
            .iter()
            .filter_map(|p| p.0.strip_prefix(&cx.shared.src_root).ok())
        {
            let mut h = &mut hierarchy;
            let mut elems = source
                .components()
                .filter_map(|s| match s {
                    Component::Normal(s) => Some(s.to_owned()),
                    _ => None,
                })
                .peekable();
            loop {
                let cur_elem = elems.next().expect("empty file path");
                if elems.peek().is_none() {
                    h.elems.insert(cur_elem);
                    break;
                } else {
                    let e = cur_elem.clone();
                    h = h.children.entry(cur_elem.clone()).or_insert_with(|| Hierarchy::new(e));
                }
            }
        }

        let dst = cx.dst.join(&format!("source-files{}.js", cx.shared.resource_suffix));
        let make_sources = || {
            let (mut all_sources, _krates) =
                try_err!(collect(&dst, &krate.name(cx.tcx()).as_str(), "sourcesIndex"), &dst);
            all_sources.push(format!(
                "sourcesIndex[\"{}\"] = {};",
                &krate.name(cx.tcx()),
                hierarchy.to_json_string()
            ));
            all_sources.sort();
            Ok(format!(
                "var N = null;var sourcesIndex = {{}};\n{}\ncreateSourceSidebar();\n",
                all_sources.join("\n")
            )
            .into_bytes())
        };
        write_crate("source-files.js", &make_sources)?;
    }

    // Update the search index and crate list.
    let dst = cx.dst.join(&format!("search-index{}.js", cx.shared.resource_suffix));
    let (mut all_indexes, mut krates) =
        try_err!(collect_json(&dst, &krate.name(cx.tcx()).as_str()), &dst);
    all_indexes.push(search_index);
    krates.push(krate.name(cx.tcx()).to_string());
    krates.sort();

    // Sort the indexes by crate so the file will be generated identically even
    // with rustdoc running in parallel.
    all_indexes.sort();
    write_crate("search-index.js", &|| {
        let mut v = String::from("var searchIndex = JSON.parse('{\\\n");
        v.push_str(&all_indexes.join(",\\\n"));
        v.push_str("\\\n}');\nif (window.initSearch) {window.initSearch(searchIndex)};");
        Ok(v.into_bytes())
    })?;

    write_crate("crates.js", &|| {
        let krates = krates.iter().map(|k| format!("\"{}\"", k)).join(",");
        Ok(format!("window.ALL_CRATES = [{}];", krates).into_bytes())
    })?;

    if options.enable_index_page {
        if let Some(index_page) = options.index_page.clone() {
            let mut md_opts = options.clone();
            md_opts.output = cx.dst.clone();
            md_opts.external_html = (*cx.shared).layout.external_html.clone();

            crate::markdown::render(&index_page, md_opts, cx.shared.edition())
                .map_err(|e| Error::new(e, &index_page))?;
        } else {
            let dst = cx.dst.join("index.html");
            let page = layout::Page {
                title: "Index of crates",
                css_class: "mod",
                root_path: "./",
                static_root_path: cx.shared.static_root_path.as_deref(),
                description: "List of crates",
                keywords: BASIC_KEYWORDS,
                resource_suffix: &cx.shared.resource_suffix,
                extra_scripts: &[],
                static_extra_scripts: &[],
            };

            let content = format!(
                "<h1 class=\"fqn\">\
                     <span class=\"in-band\">List of all crates</span>\
                </h1><ul class=\"crate mod\">{}</ul>",
                krates
                    .iter()
                    .map(|s| {
                        format!(
                            "<li><a class=\"crate mod\" href=\"{}index.html\">{}</a></li>",
                            ensure_trailing_slash(s),
                            s
                        )
                    })
                    .collect::<String>()
            );
            let v = layout::render(
                &cx.shared.templates,
                &cx.shared.layout,
                &page,
                "",
                content,
                &cx.shared.style_files,
            );
            cx.shared.fs.write(dst, v)?;
        }
    }

    // Update the list of all implementors for traits
    let dst = cx.dst.join("implementors");
    let cache = cx.cache();
    for (&did, imps) in &cache.implementors {
        // Private modules can leak through to this phase of rustdoc, which
        // could contain implementations for otherwise private types. In some
        // rare cases we could find an implementation for an item which wasn't
        // indexed, so we just skip this step in that case.
        //
        // FIXME: this is a vague explanation for why this can't be a `get`, in
        //        theory it should be...
        let &(ref remote_path, remote_item_type) = match cache.paths.get(&did) {
            Some(p) => p,
            None => match cache.external_paths.get(&did) {
                Some(p) => p,
                None => continue,
            },
        };

        #[derive(Serialize)]
        struct Implementor {
            text: String,
            synthetic: bool,
            types: Vec<String>,
        }

        let implementors = imps
            .iter()
            .filter_map(|imp| {
                // If the trait and implementation are in the same crate, then
                // there's no need to emit information about it (there's inlining
                // going on). If they're in different crates then the crate defining
                // the trait will be interested in our implementation.
                //
                // If the implementation is from another crate then that crate
                // should add it.
                if imp.impl_item.def_id.krate() == did.krate || !imp.impl_item.def_id.is_local() {
                    None
                } else {
                    Some(Implementor {
                        text: imp.inner_impl().print(false, cx).to_string(),
                        synthetic: imp.inner_impl().kind.is_auto(),
                        types: collect_paths_for_type(imp.inner_impl().for_.clone(), cache),
                    })
                }
            })
            .collect::<Vec<_>>();

        // Only create a js file if we have impls to add to it. If the trait is
        // documented locally though we always create the file to avoid dead
        // links.
        if implementors.is_empty() && !cache.paths.contains_key(&did) {
            continue;
        }

        let implementors = format!(
            r#"implementors["{}"] = {};"#,
            krate.name(cx.tcx()),
            serde_json::to_string(&implementors).unwrap()
        );

        let mut mydst = dst.clone();
        for part in &remote_path[..remote_path.len() - 1] {
            mydst.push(part);
        }
        cx.shared.ensure_dir(&mydst)?;
        mydst.push(&format!("{}.{}.js", remote_item_type, remote_path[remote_path.len() - 1]));

        let (mut all_implementors, _) =
            try_err!(collect(&mydst, &krate.name(cx.tcx()).as_str(), "implementors"), &mydst);
        all_implementors.push(implementors);
        // Sort the implementors by crate so the file will be generated
        // identically even with rustdoc running in parallel.
        all_implementors.sort();

        let mut v = String::from("(function() {var implementors = {};\n");
        for implementor in &all_implementors {
            writeln!(v, "{}", *implementor).unwrap();
        }
        v.push_str(
            "if (window.register_implementors) {\
                 window.register_implementors(implementors);\
             } else {\
                 window.pending_implementors = implementors;\
             }",
        );
        v.push_str("})()");
        cx.shared.fs.write(mydst, v)?;
    }
    Ok(())
}
