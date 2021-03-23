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
use crate::config::RenderOptions;
use crate::docfs::{DocFS, PathError};
use crate::error::Error;
use crate::formats::FormatRenderer;
use crate::html::{layout, static_files};

crate static FILES_UNVERSIONED: Lazy<FxHashMap<&str, &[u8]>> = Lazy::new(|| {
    map! {
        "FiraSans-Regular.woff2" => static_files::fira_sans::REGULAR2,
        "FiraSans-Medium.woff2" => static_files::fira_sans::MEDIUM2,
        "FiraSans-Regular.woff" => static_files::fira_sans::REGULAR,
        "FiraSans-Medium.woff" => static_files::fira_sans::MEDIUM,
        "FiraSans-LICENSE.txt" => static_files::fira_sans::LICENSE,
        "SourceSerifPro-Regular.ttf.woff" => static_files::source_serif_pro::REGULAR,
        "SourceSerifPro-Bold.ttf.woff" => static_files::source_serif_pro::BOLD,
        "SourceSerifPro-It.ttf.woff" => static_files::source_serif_pro::ITALIC,
        "SourceSerifPro-LICENSE.md" => static_files::source_serif_pro::LICENSE,
        "SourceCodePro-Regular.ttf.woff" => static_files::source_code_pro::REGULAR,
        "SourceCodePro-Semibold.ttf.woff" => static_files::source_code_pro::SEMIBOLD,
        "SourceCodePro-It.ttf.woff" => static_files::source_code_pro::ITALIC,
        "SourceCodePro-LICENSE.txt" => static_files::source_code_pro::LICENSE,
        "LICENSE-MIT.txt" => static_files::LICENSE_MIT,
        "LICENSE-APACHE.txt" => static_files::LICENSE_APACHE,
        "COPYRIGHT.txt" => static_files::COPYRIGHT,
    }
});

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

    // Add all the static files. These may already exist, but we just
    // overwrite them anyway to make sure that they're fresh and up-to-date.

    write_minify(
        &cx.shared.fs,
        cx.path("rustdoc.css"),
        static_files::RUSTDOC_CSS,
        options.enable_minification,
    )?;
    write_minify(
        &cx.shared.fs,
        cx.path("settings.css"),
        static_files::SETTINGS_CSS,
        options.enable_minification,
    )?;
    write_minify(
        &cx.shared.fs,
        cx.path("noscript.css"),
        static_files::NOSCRIPT_CSS,
        options.enable_minification,
    )?;

    // To avoid "light.css" to be overwritten, we'll first run over the received themes and only
    // then we'll run over the "official" styles.
    let mut themes: FxHashSet<String> = FxHashSet::default();

    for entry in &cx.shared.style_files {
        let theme = try_none!(try_none!(entry.path.file_stem(), &entry.path).to_str(), &entry.path);
        let extension =
            try_none!(try_none!(entry.path.extension(), &entry.path).to_str(), &entry.path);

        // Handle the official themes
        match theme {
            "light" => write_minify(
                &cx.shared.fs,
                cx.path("light.css"),
                static_files::themes::LIGHT,
                options.enable_minification,
            )?,
            "dark" => write_minify(
                &cx.shared.fs,
                cx.path("dark.css"),
                static_files::themes::DARK,
                options.enable_minification,
            )?,
            "ayu" => write_minify(
                &cx.shared.fs,
                cx.path("ayu.css"),
                static_files::themes::AYU,
                options.enable_minification,
            )?,
            _ => {
                // Handle added third-party themes
                let content = try_err!(fs::read(&entry.path), &entry.path);
                cx.shared
                    .fs
                    .write(cx.path(&format!("{}.{}", theme, extension)), content.as_slice())?;
            }
        };

        themes.insert(theme.to_owned());
    }

    let write = |p, c| cx.shared.fs.write(p, c);
    if (*cx.shared).layout.logo.is_empty() {
        write(cx.path("rust-logo.png"), static_files::RUST_LOGO)?;
    }
    if (*cx.shared).layout.favicon.is_empty() {
        write(cx.path("favicon.svg"), static_files::RUST_FAVICON_SVG)?;
        write(cx.path("favicon-16x16.png"), static_files::RUST_FAVICON_PNG_16)?;
        write(cx.path("favicon-32x32.png"), static_files::RUST_FAVICON_PNG_32)?;
    }
    write(cx.path("brush.svg"), static_files::BRUSH_SVG)?;
    write(cx.path("wheel.svg"), static_files::WHEEL_SVG)?;
    write(cx.path("down-arrow.svg"), static_files::DOWN_ARROW_SVG)?;

    let mut themes: Vec<&String> = themes.iter().collect();
    themes.sort();

    write_minify(
        &cx.shared.fs,
        cx.path("main.js"),
        &static_files::MAIN_JS.replace(
            "/* INSERT THEMES HERE */",
            &format!(" = {}", serde_json::to_string(&themes).unwrap()),
        ),
        options.enable_minification,
    )?;
    write_minify(
        &cx.shared.fs,
        cx.path("settings.js"),
        static_files::SETTINGS_JS,
        options.enable_minification,
    )?;
    if cx.shared.include_sources {
        write_minify(
            &cx.shared.fs,
            cx.path("source-script.js"),
            static_files::sidebar::SOURCE_SCRIPT,
            options.enable_minification,
        )?;
    }

    {
        write_minify(
            &cx.shared.fs,
            cx.path("storage.js"),
            &format!(
                "var resourcesSuffix = \"{}\";{}",
                cx.shared.resource_suffix,
                static_files::STORAGE_JS
            ),
            options.enable_minification,
        )?;
    }

    if let Some(ref css) = cx.shared.layout.css_file_extension {
        let out = cx.path("theme.css");
        let buffer = try_err!(fs::read_to_string(css), css);
        if !options.enable_minification {
            cx.shared.fs.write(&out, &buffer)?;
        } else {
            write_minify(&cx.shared.fs, out, &buffer, options.enable_minification)?;
        }
    }
    write_minify(
        &cx.shared.fs,
        cx.path("normalize.css"),
        static_files::NORMALIZE_CSS,
        options.enable_minification,
    )?;
    for (file, contents) in &*FILES_UNVERSIONED {
        write(cx.dst.join(file), contents)?;
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

    if cx.shared.include_sources {
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
        let (mut all_sources, _krates) =
            try_err!(collect(&dst, &krate.name.as_str(), "sourcesIndex"), &dst);
        all_sources.push(format!(
            "sourcesIndex[\"{}\"] = {};",
            &krate.name,
            hierarchy.to_json_string()
        ));
        all_sources.sort();
        let v = format!(
            "var N = null;var sourcesIndex = {{}};\n{}\ncreateSourceSidebar();\n",
            all_sources.join("\n")
        );
        cx.shared.fs.write(&dst, v.as_bytes())?;
    }

    // Update the search index and crate list.
    let dst = cx.dst.join(&format!("search-index{}.js", cx.shared.resource_suffix));
    let (mut all_indexes, mut krates) = try_err!(collect_json(&dst, &krate.name.as_str()), &dst);
    all_indexes.push(search_index);
    krates.push(krate.name.to_string());
    krates.sort();

    // Sort the indexes by crate so the file will be generated identically even
    // with rustdoc running in parallel.
    all_indexes.sort();
    {
        let mut v = String::from("var searchIndex = JSON.parse('{\\\n");
        v.push_str(&all_indexes.join(",\\\n"));
        v.push_str("\\\n}');\ninitSearch(searchIndex);");
        cx.shared.fs.write(&dst, &v)?;
    }

    let crate_list_dst = cx.dst.join(&format!("crates{}.js", cx.shared.resource_suffix));
    let crate_list =
        format!("window.ALL_CRATES = [{}];", krates.iter().map(|k| format!("\"{}\"", k)).join(","));
    cx.shared.fs.write(&crate_list_dst, &crate_list)?;

    if options.enable_index_page {
        if let Some(index_page) = options.index_page.clone() {
            let mut md_opts = options.clone();
            md_opts.output = cx.dst.clone();
            md_opts.external_html = (*cx.shared).layout.external_html.clone();

            crate::markdown::render(&index_page, md_opts, cx.shared.edition)
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
            let v = layout::render(&cx.shared.layout, &page, "", content, &cx.shared.style_files);
            cx.shared.fs.write(&dst, v.as_bytes())?;
        }
    }

    // Update the list of all implementors for traits
    let dst = cx.dst.join("implementors");
    for (&did, imps) in &cx.cache.implementors {
        // Private modules can leak through to this phase of rustdoc, which
        // could contain implementations for otherwise private types. In some
        // rare cases we could find an implementation for an item which wasn't
        // indexed, so we just skip this step in that case.
        //
        // FIXME: this is a vague explanation for why this can't be a `get`, in
        //        theory it should be...
        let &(ref remote_path, remote_item_type) = match cx.cache.paths.get(&did) {
            Some(p) => p,
            None => match cx.cache.external_paths.get(&did) {
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
                if imp.impl_item.def_id.krate == did.krate || !imp.impl_item.def_id.is_local() {
                    None
                } else {
                    Some(Implementor {
                        text: imp.inner_impl().print(cx.cache(), false).to_string(),
                        synthetic: imp.inner_impl().synthetic,
                        types: collect_paths_for_type(imp.inner_impl().for_.clone(), cx.cache()),
                    })
                }
            })
            .collect::<Vec<_>>();

        // Only create a js file if we have impls to add to it. If the trait is
        // documented locally though we always create the file to avoid dead
        // links.
        if implementors.is_empty() && !cx.cache.paths.contains_key(&did) {
            continue;
        }

        let implementors = format!(
            r#"implementors["{}"] = {};"#,
            krate.name,
            serde_json::to_string(&implementors).unwrap()
        );

        let mut mydst = dst.clone();
        for part in &remote_path[..remote_path.len() - 1] {
            mydst.push(part);
        }
        cx.shared.ensure_dir(&mydst)?;
        mydst.push(&format!("{}.{}.js", remote_item_type, remote_path[remote_path.len() - 1]));

        let (mut all_implementors, _) =
            try_err!(collect(&mydst, &krate.name.as_str(), "implementors"), &mydst);
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
        cx.shared.fs.write(&mydst, &v)?;
    }
    Ok(())
}

fn write_minify(
    fs: &DocFS,
    dst: PathBuf,
    contents: &str,
    enable_minification: bool,
) -> Result<(), Error> {
    if enable_minification {
        if dst.extension() == Some(&OsStr::new("css")) {
            let res = try_none!(minifier::css::minify(contents).ok(), &dst);
            fs.write(dst, res.as_bytes())
        } else {
            fs.write(dst, minifier::js::minify(contents).as_bytes())
        }
    } else {
        fs.write(dst, contents.as_bytes())
    }
}
