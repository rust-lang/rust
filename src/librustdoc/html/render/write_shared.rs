use std::cell::RefCell;
use std::fs::{self, File};
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::path::{Component, Path};
use std::rc::{Rc, Weak};

use indexmap::IndexMap;
use itertools::Itertools;
use rustc_data_structures::flock;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams};
use rustc_span::def_id::DefId;
use rustc_span::Symbol;
use serde::ser::SerializeSeq;
use serde::{Serialize, Serializer};

use super::{collect_paths_for_type, ensure_trailing_slash, Context, RenderMode};
use crate::clean::{Crate, Item, ItemId, ItemKind};
use crate::config::{EmitType, RenderOptions};
use crate::docfs::PathError;
use crate::error::Error;
use crate::formats::cache::Cache;
use crate::formats::item_type::ItemType;
use crate::formats::Impl;
use crate::html::format::Buffer;
use crate::html::render::{AssocItemLink, ImplRenderingParameters};
use crate::html::{layout, static_files};
use crate::visit::DocVisitor;
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
        let filename = format!("{theme}{suffix}.{extension}", suffix = cx.shared.resource_suffix);
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
            let prefix = format!("\"{krate}\"");
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

    /// Read a file and return all lines that match the <code>"{crate}":{data},\ </code> format,
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
            let prefix = format!("[\"{krate}\"");
            for line in BufReader::new(File::open(path)?).lines() {
                let line = line?;
                if !line.starts_with("[\"") {
                    continue;
                }
                if line.starts_with(&prefix) {
                    continue;
                }
                if line.ends_with("],\\") {
                    ret.push(line[..line.len() - 2].to_string());
                } else {
                    // Ends with "\\" (it's the case for the last added crate line)
                    ret.push(line[..line.len() - 1].to_string());
                }
                krates.push(
                    line[1..] // We skip the `[` parent at the beginning of the line.
                        .split('"')
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
                format!(",[{subs}]")
            };
            let files = files.join(",");
            let files = if files.is_empty() { String::new() } else { format!(",[{files}]") };
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
        let dst = cx.dst.join(&format!("src-files{}.js", cx.shared.resource_suffix));
        let make_sources = || {
            let (mut all_sources, _krates) =
                try_err!(collect_json(&dst, krate.name(cx.tcx()).as_str()), &dst);
            all_sources.push(format!(
                r#"["{}",{}]"#,
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
            // This needs to be `var`, not `const`.
            // This variable needs declared in the current global scope so that if
            // src-script.js loads first, it can pick it up.
            let mut v = String::from("var srcIndex = new Map(JSON.parse('[\\\n");
            v.push_str(&all_sources.join(",\\\n"));
            v.push_str("\\\n]'));\ncreateSrcSidebar();\n");
            Ok(v.into_bytes())
        };
        write_invocation_specific("src-files.js", &make_sources)?;
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
        // This needs to be `var`, not `const`.
        // This variable needs declared in the current global scope so that if
        // search.js loads first, it can pick it up.
        let mut v = String::from("var searchIndex = new Map(JSON.parse('[\\\n");
        v.push_str(&all_indexes.join(",\\\n"));
        v.push_str(
            r#"\
]'));
if (typeof exports !== 'undefined') exports.searchIndex = searchIndex;
else if (window.initSearch) window.initSearch(searchIndex);
"#,
        );
        Ok(v.into_bytes())
    })?;

    write_invocation_specific("crates.js", &|| {
        let krates = krates.iter().map(|k| format!("\"{k}\"")).join(",");
        Ok(format!("window.ALL_CRATES = [{krates}];").into_bytes())
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
                css_class: "mod sys",
                root_path: "./",
                static_root_path: shared.static_root_path.as_deref(),
                description: "List of crates",
                resource_suffix: &shared.resource_suffix,
                rust_logo: true,
            };

            let content = format!(
                "<h1>List of all crates</h1><ul class=\"all-items\">{}</ul>",
                krates.iter().format_with("", |k, f| {
                    f(&format_args!(
                        "<li><a href=\"{trailing_slash}index.html\">{k}</a></li>",
                        trailing_slash = ensure_trailing_slash(k),
                    ))
                })
            );
            let v = layout::render(&shared.layout, &page, "", content, &shared.style_files);
            shared.fs.write(dst, v)?;
        }
    }

    let cloned_shared = Rc::clone(&cx.shared);
    let cache = &cloned_shared.cache;

    // Collect the list of aliased types and their aliases.
    // <https://github.com/search?q=repo%3Arust-lang%2Frust+[RUSTDOCIMPL]+type.impl&type=code>
    //
    // The clean AST has type aliases that point at their types, but
    // this visitor works to reverse that: `aliased_types` is a map
    // from target to the aliases that reference it, and each one
    // will generate one file.
    struct TypeImplCollector<'cx, 'cache> {
        // Map from DefId-of-aliased-type to its data.
        aliased_types: IndexMap<DefId, AliasedType<'cache>>,
        visited_aliases: FxHashSet<DefId>,
        cache: &'cache Cache,
        cx: &'cache mut Context<'cx>,
    }
    // Data for an aliased type.
    //
    // In the final file, the format will be roughly:
    //
    // ```json
    // // type.impl/CRATE/TYPENAME.js
    // JSONP(
    // "CRATE": [
    //   ["IMPL1 HTML", "ALIAS1", "ALIAS2", ...],
    //   ["IMPL2 HTML", "ALIAS3", "ALIAS4", ...],
    //    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ struct AliasedType
    //   ...
    // ]
    // )
    // ```
    struct AliasedType<'cache> {
        // This is used to generate the actual filename of this aliased type.
        target_fqp: &'cache [Symbol],
        target_type: ItemType,
        // This is the data stored inside the file.
        // ItemId is used to deduplicate impls.
        impl_: IndexMap<ItemId, AliasedTypeImpl<'cache>>,
    }
    // The `impl_` contains data that's used to figure out if an alias will work,
    // and to generate the HTML at the end.
    //
    // The `type_aliases` list is built up with each type alias that matches.
    struct AliasedTypeImpl<'cache> {
        impl_: &'cache Impl,
        type_aliases: Vec<(&'cache [Symbol], Item)>,
    }
    impl<'cx, 'cache> DocVisitor for TypeImplCollector<'cx, 'cache> {
        fn visit_item(&mut self, it: &Item) {
            self.visit_item_recur(it);
            let cache = self.cache;
            let ItemKind::TypeAliasItem(ref t) = *it.kind else { return };
            let Some(self_did) = it.item_id.as_def_id() else { return };
            if !self.visited_aliases.insert(self_did) {
                return;
            }
            let Some(target_did) = t.type_.def_id(cache) else { return };
            let get_extern = { || cache.external_paths.get(&target_did) };
            let Some(&(ref target_fqp, target_type)) =
                cache.paths.get(&target_did).or_else(get_extern)
            else {
                return;
            };
            let aliased_type = self.aliased_types.entry(target_did).or_insert_with(|| {
                let impl_ = cache
                    .impls
                    .get(&target_did)
                    .map(|v| &v[..])
                    .unwrap_or_default()
                    .iter()
                    .map(|impl_| {
                        (
                            impl_.impl_item.item_id,
                            AliasedTypeImpl { impl_, type_aliases: Vec::new() },
                        )
                    })
                    .collect();
                AliasedType { target_fqp: &target_fqp[..], target_type, impl_ }
            });
            let get_local = { || cache.paths.get(&self_did).map(|(p, _)| p) };
            let Some(self_fqp) = cache.exact_paths.get(&self_did).or_else(get_local) else {
                return;
            };
            let aliased_ty = self.cx.tcx().type_of(self_did).skip_binder();
            // Exclude impls that are directly on this type. They're already in the HTML.
            // Some inlining scenarios can cause there to be two versions of the same
            // impl: one on the type alias and one on the underlying target type.
            let mut seen_impls: FxHashSet<ItemId> = cache
                .impls
                .get(&self_did)
                .map(|s| &s[..])
                .unwrap_or_default()
                .iter()
                .map(|i| i.impl_item.item_id)
                .collect();
            for (impl_item_id, aliased_type_impl) in &mut aliased_type.impl_ {
                // Only include this impl if it actually unifies with this alias.
                // Synthetic impls are not included; those are also included in the HTML.
                //
                // FIXME(lazy_type_alias): Once the feature is complete or stable, rewrite this
                // to use type unification.
                // Be aware of `tests/rustdoc/type-alias/deeply-nested-112515.rs` which might regress.
                let Some(impl_did) = impl_item_id.as_def_id() else { continue };
                let for_ty = self.cx.tcx().type_of(impl_did).skip_binder();
                let reject_cx =
                    DeepRejectCtxt { treat_obligation_params: TreatParams::AsCandidateKey };
                if !reject_cx.types_may_unify(aliased_ty, for_ty) {
                    continue;
                }
                // Avoid duplicates
                if !seen_impls.insert(*impl_item_id) {
                    continue;
                }
                // This impl was not found in the set of rejected impls
                aliased_type_impl.type_aliases.push((&self_fqp[..], it.clone()));
            }
        }
    }
    let mut type_impl_collector = TypeImplCollector {
        aliased_types: IndexMap::default(),
        visited_aliases: FxHashSet::default(),
        cache,
        cx,
    };
    DocVisitor::visit_crate(&mut type_impl_collector, &krate);
    // Final serialized form of the alias impl
    struct AliasSerializableImpl {
        text: String,
        trait_: Option<String>,
        aliases: Vec<String>,
    }
    impl Serialize for AliasSerializableImpl {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut seq = serializer.serialize_seq(None)?;
            seq.serialize_element(&self.text)?;
            if let Some(trait_) = &self.trait_ {
                seq.serialize_element(trait_)?;
            } else {
                seq.serialize_element(&0)?;
            }
            for type_ in &self.aliases {
                seq.serialize_element(type_)?;
            }
            seq.end()
        }
    }
    let cx = type_impl_collector.cx;
    let dst = cx.dst.join("type.impl");
    let aliased_types = type_impl_collector.aliased_types;
    for aliased_type in aliased_types.values() {
        let impls = aliased_type
            .impl_
            .values()
            .flat_map(|AliasedTypeImpl { impl_, type_aliases }| {
                let mut ret = Vec::new();
                let trait_ = impl_
                    .inner_impl()
                    .trait_
                    .as_ref()
                    .map(|trait_| format!("{:#}", trait_.print(cx)));
                // render_impl will filter out "impossible-to-call" methods
                // to make that functionality work here, it needs to be called with
                // each type alias, and if it gives a different result, split the impl
                for &(type_alias_fqp, ref type_alias_item) in type_aliases {
                    let mut buf = Buffer::html();
                    cx.id_map = Default::default();
                    cx.deref_id_map = Default::default();
                    let target_did = impl_
                        .inner_impl()
                        .trait_
                        .as_ref()
                        .map(|trait_| trait_.def_id())
                        .or_else(|| impl_.inner_impl().for_.def_id(cache));
                    let provided_methods;
                    let assoc_link = if let Some(target_did) = target_did {
                        provided_methods = impl_.inner_impl().provided_trait_methods(cx.tcx());
                        AssocItemLink::GotoSource(ItemId::DefId(target_did), &provided_methods)
                    } else {
                        AssocItemLink::Anchor(None)
                    };
                    super::render_impl(
                        &mut buf,
                        cx,
                        *impl_,
                        &type_alias_item,
                        assoc_link,
                        RenderMode::Normal,
                        None,
                        &[],
                        ImplRenderingParameters {
                            show_def_docs: true,
                            show_default_items: true,
                            show_non_assoc_items: true,
                            toggle_open_by_default: true,
                        },
                    );
                    let text = buf.into_inner();
                    let type_alias_fqp = (*type_alias_fqp).iter().join("::");
                    if Some(&text) == ret.last().map(|s: &AliasSerializableImpl| &s.text) {
                        ret.last_mut()
                            .expect("already established that ret.last() is Some()")
                            .aliases
                            .push(type_alias_fqp);
                    } else {
                        ret.push(AliasSerializableImpl {
                            text,
                            trait_: trait_.clone(),
                            aliases: vec![type_alias_fqp],
                        })
                    }
                }
                ret
            })
            .collect::<Vec<_>>();

        // FIXME: this fixes only rustdoc part of instability of trait impls
        // for js files, see #120371
        // Manually collect to string and sort to make list not depend on order
        let mut impls = impls
            .iter()
            .map(|i| serde_json::to_string(i).expect("failed serde conversion"))
            .collect::<Vec<_>>();
        impls.sort();

        let impls = format!(r#""{}":[{}]"#, krate.name(cx.tcx()), impls.join(","));

        let mut mydst = dst.clone();
        for part in &aliased_type.target_fqp[..aliased_type.target_fqp.len() - 1] {
            mydst.push(part.to_string());
        }
        cx.shared.ensure_dir(&mydst)?;
        let aliased_item_type = aliased_type.target_type;
        mydst.push(&format!(
            "{aliased_item_type}.{}.js",
            aliased_type.target_fqp[aliased_type.target_fqp.len() - 1]
        ));

        let (mut all_impls, _) = try_err!(collect(&mydst, krate.name(cx.tcx()).as_str()), &mydst);
        all_impls.push(impls);
        // Sort the implementors by crate so the file will be generated
        // identically even with rustdoc running in parallel.
        all_impls.sort();

        let mut v = String::from("(function() {var type_impls = {\n");
        v.push_str(&all_impls.join(",\n"));
        v.push_str("\n};");
        v.push_str(
            "if (window.register_type_impls) {\
                 window.register_type_impls(type_impls);\
             } else {\
                 window.pending_type_impls = type_impls;\
             }",
        );
        v.push_str("})()");
        cx.shared.fs.write(mydst, v)?;
    }

    // Update the list of all implementors for traits
    // <https://github.com/search?q=repo%3Arust-lang%2Frust+[RUSTDOCIMPL]+trait.impl&type=code>
    let dst = cx.dst.join("trait.impl");
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

        // FIXME: this fixes only rustdoc part of instability of trait impls
        // for js files, see #120371
        // Manually collect to string and sort to make list not depend on order
        let mut implementors = implementors
            .iter()
            .map(|i| serde_json::to_string(i).expect("failed serde conversion"))
            .collect::<Vec<_>>();
        implementors.sort();

        let implementors = format!(r#""{}":[{}]"#, krate.name(cx.tcx()), implementors.join(","));

        let mut mydst = dst.clone();
        for part in &remote_path[..remote_path.len() - 1] {
            mydst.push(part.to_string());
        }
        cx.shared.ensure_dir(&mydst)?;
        mydst.push(&format!("{remote_item_type}.{}.js", remote_path[remote_path.len() - 1]));

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
