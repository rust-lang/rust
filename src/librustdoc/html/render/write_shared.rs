use std::cell::RefCell;
use std::fs::{self, File};
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::path::{Component, Path};
use std::rc::{Rc, Weak};

use itertools::Itertools;
use rustc_data_structures::flock;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use serde::ser::SerializeSeq;
use serde::{Serialize, Serializer};

use super::{collect_paths_for_type, ensure_trailing_slash, Context};
use crate::clean::Crate;
use crate::config::{EmitType, RenderOptions};
use crate::docfs::PathError;
use crate::error::Error;
use crate::html::{layout, static_files};
use crate::{try_err, try_none};

/// Rustdoc writes out two kinds of shared files:
///  - Static files, which are embedded in the rustdoc binary and are written with a
///    filename that includes a hash of their contents. These will always have a new
///    URL if the contents change, so they are safe to cache with the
///    `Cache-Control: immutable` directive. They are written under the static.files/
///    directory and are written when --emit-type is empty (default) or contains
///    "toolchain-specific". If using the --static-root-path flag, it should point
///    to a URL path prefix where each of these filenames can be fetched.
///  - Invocation specific files. These are generated based on the crate(s) being
///    documented. Their filenames need to be predictable without knowing their
///    contents, so they do not include a hash in their filename and are not safe to
///    cache with `Cache-Control: immutable`. They include the contents of the
///    --resource-suffix flag and are emitted when --emit-type is empty (default)
///    or contains "invocation-specific".
pub(super) fn write_shared(
    cx: &mut Context<'_>,
    krate: &Crate,
    search_index: String,
    options: &RenderOptions,
) -> Result<(), Error> {
    // Write out the shared files. Note that these are shared among all rustdoc
    // docs placed in the output directory, so this needs to be a synchronized
    // operation with respect to all other rustdocs running around.
    let lock_file = cx.dst.join(".lock");
    let _lock = try_err!(flock::Lock::new(&lock_file, true, true, true), &lock_file);

    // InvocationSpecific resources should always be dynamic.
    let write_invocation_specific = |p: &str, make_content: &dyn Fn() -> Result<Vec<u8>, Error>| {
        let content = make_content()?;
        if options.emit.is_empty() || options.emit.contains(&EmitType::InvocationSpecific) {
            let output_filename = static_files::suffix_path(p, &cx.shared.resource_suffix);
            cx.shared.fs.write(cx.dst.join(output_filename), content)
        } else {
            Ok(())
        }
    };

    cx.shared
        .fs
        .create_dir_all(cx.dst.join("static.files"))
        .map_err(|e| PathError::new(e, "static.files"))?;

    // Handle added third-party themes
    for entry in &cx.shared.style_files {
        let theme = entry.basename()?;
        let extension =
            try_none!(try_none!(entry.path.extension(), &entry.path).to_str(), &entry.path);

        // Skip the official themes. They are written below as part of STATIC_FILES_LIST.
        if matches!(theme.as_str(), "light" | "dark" | "ayu") {
            continue;
        }

        let bytes = try_err!(fs::read(&entry.path), &entry.path);
        let filename = format!("{}{}.{}", theme, cx.shared.resource_suffix, extension);
        cx.shared.fs.write(cx.dst.join(filename), bytes)?;
    }

    // When the user adds their own CSS files with --extend-css, we write that as an
    // invocation-specific file (that is, with a resource suffix).
    if let Some(ref css) = cx.shared.layout.css_file_extension {
        let buffer = try_err!(fs::read_to_string(css), css);
        let path = static_files::suffix_path("theme.css", &cx.shared.resource_suffix);
        cx.shared.fs.write(cx.dst.join(path), buffer)?;
    }

    if options.emit.is_empty() || options.emit.contains(&EmitType::Toolchain) {
        let static_dir = cx.dst.join(Path::new("static.files"));
        static_files::for_each(|f: &static_files::StaticFile| {
            let filename = static_dir.join(f.output_filename());
            cx.shared.fs.write(filename, f.minified())
        })?;
    }

    /// Read a file and return all lines that match the `"{crate}":{data},` format,
    /// and return a tuple `(Vec<DataString>, Vec<CrateNameString>)`.
    ///
    /// This forms the payload of files that look like this:
    ///
    /// ```javascript
    /// var data = {
    /// "{crate1}":{data},
    /// "{crate2}":{data}
    /// };
    /// use_data(data);
    /// ```
    ///
    /// The file needs to be formatted so that *only crate data lines start with `"`*.
    fn collect(path: &Path, krate: &str) -> io::Result<(Vec<String>, Vec<String>)> {
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
                if line.ends_with(',') {
                    ret.push(line[..line.len() - 1].to_string());
                } else {
                    // No comma (it's the case for the last added crate line)
                    ret.push(line.to_string());
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

    /// Read a file and return all lines that match the <code>"{crate}":{data},\</code> format,
    /// and return a tuple `(Vec<DataString>, Vec<CrateNameString>)`.
    ///
    /// This forms the payload of files that look like this:
    ///
    /// ```javascript
    /// var data = JSON.parse('{\
    /// "{crate1}":{data},\
    /// "{crate2}":{data}\
    /// }');
    /// use_data(data);
    /// ```
    ///
    /// The file needs to be formatted so that *only crate data lines start with `"`*.
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

    #[derive(Debug, Default)]
    struct Hierarchy {
        parent: Weak<Self>,
        elem: OsString,
        children: RefCell<FxHashMap<OsString, Rc<Self>>>,
        elems: RefCell<FxHashSet<OsString>>,
    }

    impl Hierarchy {
        fn with_parent(elem: OsString, parent: &Rc<Self>) -> Self {
            Self { elem, parent: Rc::downgrade(parent), ..Self::default() }
        }

        fn to_json_string(&self) -> String {
            let borrow = self.children.borrow();
            let mut subs: Vec<_> = borrow.values().collect();
            subs.sort_unstable_by(|a, b| a.elem.cmp(&b.elem));
            let mut files = self
                .elems
                .borrow()
                .iter()
                .map(|s| format!("\"{}\"", s.to_str().expect("invalid osstring conversion")))
                .collect::<Vec<_>>();
            files.sort_unstable();
            let subs = subs.iter().map(|s| s.to_json_string()).collect::<Vec<_>>().join(",");
            let dirs = if subs.is_empty() && files.is_empty() {
                String::new()
            } else {
                format!(",[{}]", subs)
            };
            let files = files.join(",");
            let files = if files.is_empty() { String::new() } else { format!(",[{}]", files) };
            format!(
                "[\"{name}\"{dirs}{files}]",
                name = self.elem.to_str().expect("invalid osstring conversion"),
                dirs = dirs,
                files = files
            )
        }

        fn add_path(self: &Rc<Self>, path: &Path) {
            let mut h = Rc::clone(&self);
            let mut elems = path
                .components()
                .filter_map(|s| match s {
                    Component::Normal(s) => Some(s.to_owned()),
                    Component::ParentDir => Some(OsString::from("..")),
                    _ => None,
                })
                .peekable();
            loop {
                let cur_elem = elems.next().expect("empty file path");
                if cur_elem == ".." {
                    if let Some(parent) = h.parent.upgrade() {
                        h = parent;
                    }
                    continue;
                }
                if elems.peek().is_none() {
                    h.elems.borrow_mut().insert(cur_elem);
                    break;
                } else {
                    let entry = Rc::clone(
                        h.children
                            .borrow_mut()
                            .entry(cur_elem.clone())
                            .or_insert_with(|| Rc::new(Self::with_parent(cur_elem, &h))),
                    );
                    h = entry;
                }
            }
        }
    }

    if cx.include_sources {
        let hierarchy = Rc::new(Hierarchy::default());
        for source in cx
            .shared
            .local_sources
            .iter()
            .filter_map(|p| p.0.strip_prefix(&cx.shared.src_root).ok())
        {
            hierarchy.add_path(source);
        }
        let hierarchy = Rc::try_unwrap(hierarchy).unwrap();
        let dst = cx.dst.join(&format!("source-files{}.js", cx.shared.resource_suffix));
        let make_sources = || {
            let (mut all_sources, _krates) =
                try_err!(collect_json(&dst, krate.name(cx.tcx()).as_str()), &dst);
            all_sources.push(format!(
                r#""{}":{}"#,
                &krate.name(cx.tcx()),
                hierarchy
                    .to_json_string()
                    // All these `replace` calls are because we have to go through JS string for JSON content.
                    .replace('\\', r"\\")
                    .replace('\'', r"\'")
                    // We need to escape double quotes for the JSON.
                    .replace("\\\"", "\\\\\"")
            ));
            all_sources.sort();
            let mut v = String::from("var sourcesIndex = JSON.parse('{\\\n");
            v.push_str(&all_sources.join(",\\\n"));
            v.push_str("\\\n}');\ncreateSourceSidebar();\n");
            Ok(v.into_bytes())
        };
        write_invocation_specific("source-files.js", &make_sources)?;
    }

    // Update the search index and crate list.
    let dst = cx.dst.join(&format!("search-index{}.js", cx.shared.resource_suffix));
    let (mut all_indexes, mut krates) =
        try_err!(collect_json(&dst, krate.name(cx.tcx()).as_str()), &dst);
    all_indexes.push(search_index);
    krates.push(krate.name(cx.tcx()).to_string());
    krates.sort();

    // Sort the indexes by crate so the file will be generated identically even
    // with rustdoc running in parallel.
    all_indexes.sort();
    write_invocation_specific("search-index.js", &|| {
        let mut v = String::from("var searchIndex = JSON.parse('{\\\n");
        v.push_str(&all_indexes.join(",\\\n"));
        v.push_str(
            r#"\
}');
if (typeof window !== 'undefined' && window.initSearch) {window.initSearch(searchIndex)};
if (typeof exports !== 'undefined') {exports.searchIndex = searchIndex};
"#,
        );
        Ok(v.into_bytes())
    })?;

    write_invocation_specific("crates.js", &|| {
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
            let shared = Rc::clone(&cx.shared);
            let dst = cx.dst.join("index.html");
            let page = layout::Page {
                title: "Index of crates",
                css_class: "mod",
                root_path: "./",
                static_root_path: shared.static_root_path.as_deref(),
                description: "List of crates",
                resource_suffix: &shared.resource_suffix,
            };

            let content = format!(
                "<h1>List of all crates</h1><ul class=\"all-items\">{}</ul>",
                krates
                    .iter()
                    .map(|s| {
                        format!(
                            "<li><a href=\"{}index.html\">{}</a></li>",
                            ensure_trailing_slash(s),
                            s
                        )
                    })
                    .collect::<String>()
            );
            let v = layout::render(&shared.layout, &page, "", content, &shared.style_files);
            shared.fs.write(dst, v)?;
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
        let (remote_path, remote_item_type) = match cache.exact_paths.get(&did) {
            Some(p) => match cache.paths.get(&did).or_else(|| cache.external_paths.get(&did)) {
                Some((_, t)) => (p, t),
                None => continue,
            },
            None => match cache.external_paths.get(&did) {
                Some((p, t)) => (p, t),
                None => continue,
            },
        };

        struct Implementor {
            text: String,
            synthetic: bool,
            types: Vec<String>,
        }

        impl Serialize for Implementor {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                let mut seq = serializer.serialize_seq(None)?;
                seq.serialize_element(&self.text)?;
                if self.synthetic {
                    seq.serialize_element(&1)?;
                    seq.serialize_element(&self.types)?;
                }
                seq.end()
            }
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
                if imp.impl_item.item_id.krate() == did.krate || !imp.impl_item.item_id.is_local() {
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
            r#""{}":{}"#,
            krate.name(cx.tcx()),
            serde_json::to_string(&implementors).expect("failed serde conversion"),
        );

        let mut mydst = dst.clone();
        for part in &remote_path[..remote_path.len() - 1] {
            mydst.push(part.to_string());
        }
        cx.shared.ensure_dir(&mydst)?;
        mydst.push(&format!("{}.{}.js", remote_item_type, remote_path[remote_path.len() - 1]));

        let (mut all_implementors, _) =
            try_err!(collect(&mydst, krate.name(cx.tcx()).as_str()), &mydst);
        all_implementors.push(implementors);
        // Sort the implementors by crate so the file will be generated
        // identically even with rustdoc running in parallel.
        all_implementors.sort();

        let mut v = String::from("(function() {var implementors = {\n");
        v.push_str(&all_implementors.join(",\n"));
        v.push_str("\n};");
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
