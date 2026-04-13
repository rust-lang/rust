//! Injects Markdown (from `#[safety::requires(...)]` + `--safety-spec`) into item docs before
//! intra-doc link resolution.
//!
//! Safety tags support the following shapes:
//! - `Tag(arg1, arg2, …)` — expand using `[tag.Tag]` from the TOML.
//! - `Tag = "…"` — use the string literal as the description for this tag at this site.

use std::sync::Arc;

use regex::Regex;
use rustc_ast as ast;
use rustc_ast::token::DocFragmentKind;
use rustc_ast_pretty::pprust::{meta_list_item_to_string, path_to_string};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::DiagCtxtHandle;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_resolve::rustdoc::DocFragment;
use rustc_span::DUMMY_SP;
use rustc_span::symbol::{Symbol, sym};
use tracing::debug;

use crate::clean::{Attributes, Crate, Item, ItemKind};
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::passes::Pass;

pub(crate) const INJECT_SAFETY_DOCS: Pass = Pass {
    name: "inject-safety-docs",
    run: Some(inject_safety_docs),
    description: "injects `#[safety::requires]` text from a TOML spec into the item docs",
};

/// The safety spec.
///
/// This is a map of tag names to tag definitions.
#[derive(Debug)]
pub(crate) struct SafetySpec {
    tags: FxHashMap<String, TagDef>,
}

/// A tag definition.
///
/// The `args` are the arguments of the tag, and the `desc` is the description of the tag.
#[derive(Debug)]
struct TagDef {
    args: Vec<String>,
    desc: String,
}

/// Loads the safety spec from the given path.
///
/// The spec is a TOML file with the following structure:
///
/// ```toml
/// package.name = "my-package"
///
/// [tag.*]
/// args = ["arg1", "arg2"]
/// desc = "This is a description of the tag, containing {arg1} and {arg2}."
/// ```
///
/// The following conditions will return `None`:
///
/// * The file could not be read or parsed.
/// * The `package.name` does not match the documented crate.
/// * The `[tag.*]` table is missing.
/// * The `[tag.*]` table has no valid entries.
pub(crate) fn load_safety_spec(
    path: &std::path::Path,
    crate_name: &str,
    dcx: DiagCtxtHandle<'_>,
) -> Option<Arc<SafetySpec>> {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            dcx.struct_warn(format!("could not read `--safety-spec` file {}: {e}", path.display()))
                .emit();
            return None;
        }
    };
    let value: toml::Value = match toml::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            dcx.struct_warn(format!(
                "could not parse `--safety-spec` file {}: {e}",
                path.display()
            ))
            .emit();
            return None;
        }
    };
    let pkg_name = value
        .get("package")
        .and_then(|v| v.as_table())
        .and_then(|t| t.get("name"))
        .and_then(|v| v.as_str())?;
    if pkg_name != crate_name {
        debug!(
            "safety-spec `package.name` ({pkg_name}) does not match documented crate ({crate_name}); skipping"
        );
        return None;
    }
    let Some(tag_root) = value.get("tag").and_then(|v| v.as_table()) else {
        dcx.struct_warn(format!("`--safety-spec` {}: missing `[tag.*]` tables", path.display()))
            .emit();
        return None;
    };
    let mut tags = FxHashMap::default();
    for (tag_name, tag_v) in tag_root {
        let Some(t) = tag_v.as_table() else { continue };
        let args = t
            .get("args")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let Some(desc) = t.get("desc").and_then(|v| v.as_str()) else { continue };
        tags.insert(tag_name.clone(), TagDef { args, desc: desc.to_string() });
    }
    if tags.is_empty() {
        dcx.struct_warn(format!("`--safety-spec` {}: no valid `[tag.*]` entries", path.display()))
            .emit();
        return None;
    }
    Some(Arc::new(SafetySpec { tags }))
}

fn requires_sym() -> Symbol {
    // Common enough that repeated `intern` is fine (Symbol interner dedupes).
    Symbol::intern("requires")
}

/// The type of a tag inside `#[safety::requires(...)]`: either from TOML config or customized tags with inline text.
#[derive(Debug, Clone)]
enum SafetyTagType {
    /// `Tag(a, b)` — description comes from TOML `[tag.Tag]` `desc` with arguments filled in.
    FromConfig { args: Vec<String> },
    /// `Tag = "…"` — customized tag with inline text as the full description.
    Inline { text: String },
}

/// Parse a `#[safety::requires(...)]` attribute.
///
/// Returns `None` if the attribute is not a `#[safety::requires(...)]` attribute.
///
/// Otherwise, this will parse the attribute, returning a vector of `(tag name, tag type)` pairs.
fn parse_safety_requires(attr: &hir::Attribute) -> Option<Vec<(String, SafetyTagType)>> {
    use rustc_ast::attr::AttributeExt;

    // Check if the attribute is a `#[safety::requires(...)]` attribute.
    if !AttributeExt::path_matches(attr, &[sym::safety, requires_sym()]) {
        return None;
    }
    let list = AttributeExt::meta_item_list(attr)?;
    let mut out = Vec::new();
    // Parse the arguments to the tag.
    for inner in list {
        match inner {
            ast::MetaItemInner::MetaItem(mi) => {
                let tag_name = path_to_string(&mi.path);
                match &mi.kind {
                    ast::MetaItemKind::List(values) => {
                        let args = values.iter().map(|i| meta_list_item_to_string(i)).collect();
                        out.push((tag_name, SafetyTagType::FromConfig { args }));
                    }
                    ast::MetaItemKind::Word => {
                        out.push((tag_name, SafetyTagType::FromConfig { args: vec![] }));
                    }
                    ast::MetaItemKind::NameValue(lit) => {
                        let Some(sym) = lit.value_str() else {
                            continue;
                        };
                        out.push((tag_name, SafetyTagType::Inline { text: sym.to_string() }));
                    }
                }
            }
            ast::MetaItemInner::Lit(_) => return None,
        }
    }
    Some(out)
}

/// Collect all safety properties from the given attributes.
fn collect_safety_properties(attrs: &Attributes) -> Vec<(String, SafetyTagType)> {
    let mut props = Vec::new();
    for a in attrs.other_attrs.iter() {
        if let Some(p) = parse_safety_requires(a) {
            props.extend(p);
        }
    }
    props
}

/// Renders the description template with the given values.
///
/// # Arguments
///
/// * `def` - The tag definition.
/// * `values` - The values to substitute into the template.
///
/// # Returns
///
/// The rendered description.
///
/// # Examples
///
/// ```ignore
/// let def = TagDef { args: vec!["p", "T", "len"], desc: "pointer `{p}` must be valid for reading the `sizeof({T})* {len}` memory from it" };
/// let values = vec!["dst", "i32", "1"];
/// let rendered = render_desc(&def, &values);
/// assert_eq!(rendered, "pointer `dst` must be valid for reading the `sizeof(i32)* 1` memory from it");
/// ```
fn render_desc(def: &TagDef, values: &[String]) -> String {
    let mut rendered = def.desc.clone();
    for (i, arg_name) in def.args.iter().enumerate() {
        if let Some(value) = values.get(i) {
            rendered = rendered.replace(&format!("{{{arg_name}}}"), value);
        }
    }
    rendered
}

/// Replace all underscores (`_`) with spaces and capitalize the first letter.
fn format_customized_tag(tag: &str) -> String {
    let s = tag.replace('_', " ");
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Build the inject markdown for the given safety properties.
///
/// Returns the inject markdown.
fn build_inject_markdown(spec: &SafetySpec, props: &[(String, SafetyTagType)]) -> String {
    let mut lines = Vec::new();
    for (tag, clause) in props {
        match clause {
            SafetyTagType::Inline { text } => {
                if spec.tags.contains_key(tag) {
                    lines.push(format!("* {tag}: {text}"));
                } else {
                    let tag = format_customized_tag(tag);
                    lines.push(format!("* {tag}: {text}"));
                }
            }
            SafetyTagType::FromConfig { args } => {
                let Some(def) = spec.tags.get(tag) else {
                    continue;
                };
                let desc = render_desc(def, args);
                lines.push(format!("* {tag}: {desc}"));
            }
        }
    }
    lines.join("\n")
}

/// Inject the safety markdown into the given document.
fn inject_safety_markdown(doc: &str, inject: &str) -> String {
    if inject.is_empty() {
        return doc.to_string();
    }
    let doc = doc.trim_end();
    if doc.is_empty() {
        return format!("# Safety\n\n{inject}\n");
    }
    // Find the `# Safety` and `# Examples` headers.
    let safety_header = Regex::new(r"(?m)^#\s+Safety\s*$").unwrap();
    let examples_header = Regex::new(r"(?m)^#\s+Examples\s*$").unwrap();
    // If the `# Safety` header is found, insert the inject markdown after it.
    if let Some(m) = safety_header.find(doc) {
        let mut pos = m.end();
        if doc.as_bytes().get(pos) == Some(&b'\r') {
            pos += 1;
        }
        if doc.as_bytes().get(pos) == Some(&b'\n') {
            pos += 1;
        }
        // Blank line after the list so following paragraphs are not merged into the list
        // (CommonMark list continuation rules).
        format!("{}\n{inject}\n\n{}", &doc[..pos], &doc[pos..])
    } else if let Some(m) = examples_header.find(doc) {
        // If the `# Examples` header is found, insert the inject markdown before it.
        let insert = format!("# Safety\n\n{inject}\n\n");
        format!("{}{}{}", &doc[..m.start()], insert, &doc[m.start()..])
    } else {
        // If no `# Safety` or `# Examples` header is found, insert the inject markdown at the end of the document.
        format!("{doc}\n\n# Safety\n\n{inject}\n")
    }
}

/// Replaces the item doc with the given new doc.
fn replace_item_doc(attrs: &mut Attributes, new_doc: String, def_id: Option<DefId>) {
    let span = attrs.doc_strings.first().map(|f| f.span).unwrap_or(DUMMY_SP);
    let item_id = def_id;
    attrs.doc_strings = vec![DocFragment {
        span,
        item_id,
        doc: Symbol::intern(&new_doc),
        kind: DocFragmentKind::Raw(DUMMY_SP),
        indent: 0,
        from_expansion: attrs.doc_strings.first().map(|f| f.from_expansion).unwrap_or(false),
    }];
}

/// Checks if the given item kind is a target kind for safety documentation injection.
fn target_kind(kind: &ItemKind) -> bool {
    matches!(
        kind,
        ItemKind::FunctionItem(_)
            | ItemKind::MethodItem(..)
            | ItemKind::StructItem(_)
            | ItemKind::EnumItem(_)
            | ItemKind::UnionItem(_)
            | ItemKind::TypeAliasItem(_)
            | ItemKind::TraitItem(_)
    )
}

/// The safety injector.
///
/// This is the main struct for injecting safety documentation into the item docs.
struct SafetyInjector {
    spec: Arc<SafetySpec>,
}

impl SafetyInjector {
    /// Injects the safety documentation into the given item.
    fn inject(&self, item: &mut Item) {
        if item.is_doc_hidden() {
            return;
        }
        let kind = match &item.kind {
            ItemKind::StrippedItem(b) => b.as_ref(),
            k => k,
        };
        if !target_kind(kind) {
            return;
        }
        let props = collect_safety_properties(&item.attrs);
        if props.is_empty() {
            return;
        }
        let inject = build_inject_markdown(&self.spec, &props);
        if inject.is_empty() {
            return;
        }
        let doc = item.opt_doc_value().unwrap_or_default();
        let new_doc = inject_safety_markdown(&doc, &inject);
        let def_id = item.def_id();
        replace_item_doc(&mut item.inner.attrs, new_doc, def_id);
    }
}

impl DocFolder for SafetyInjector {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        let mut item = self.fold_item_recur(item);
        self.inject(&mut item);
        Some(item)
    }
}

/// Injects the safety documentation into the given crate.
pub(crate) fn inject_safety_docs(krate: Crate, cx: &mut DocContext<'_>) -> Crate {
    let Some(spec) = cx.safety_spec.clone() else {
        return krate;
    };
    let mut inj = SafetyInjector { spec };
    inj.fold_crate(krate)
}
