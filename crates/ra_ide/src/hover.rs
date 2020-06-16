use std::collections::{HashMap, HashSet};
use std::iter::once;

use hir::{
    db::DefDatabase, Adt, AsAssocItem, AsName, AssocItemContainer, AttrDef, Crate, Documentation,
    FieldSource, HasSource, HirDisplay, Hygiene, ItemInNs, ModPath, ModuleDef, ModuleSource,
    Semantics, Module
};
use itertools::Itertools;
use lazy_static::lazy_static;
use maplit::{hashmap, hashset};
use pulldown_cmark::{CowStr, Event, Options, Parser, Tag};
use pulldown_cmark_to_cmark::cmark;
use ra_db::SourceDatabase;
use ra_ide_db::{
    defs::{classify_name, classify_name_ref, Definition},
    RootDatabase,
};
use ra_syntax::{
    ast, ast::Path, match_ast, AstNode, SyntaxKind::*, SyntaxNode, SyntaxToken, TokenAtOffset,
};
use ra_tt::{Ident, Leaf, Literal, Punct, TokenTree};
use url::Url;

use crate::{
    display::{
        macro_label, rust_code_markup, rust_code_markup_with_doc, ShortLabel, ToNav, TryToNav,
    },
    runnables::runnable,
    FileId, FilePosition, NavigationTarget, RangeInfo, Runnable,
};
use test_utils::mark;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HoverConfig {
    pub implementations: bool,
    pub run: bool,
    pub debug: bool,
    pub goto_type_def: bool,
}

impl Default for HoverConfig {
    fn default() -> Self {
        Self { implementations: true, run: true, debug: true, goto_type_def: true }
    }
}

impl HoverConfig {
    pub const NO_ACTIONS: Self =
        Self { implementations: false, run: false, debug: false, goto_type_def: false };

    pub fn any(&self) -> bool {
        self.implementations || self.runnable() || self.goto_type_def
    }

    pub fn none(&self) -> bool {
        !self.any()
    }

    pub fn runnable(&self) -> bool {
        self.run || self.debug
    }
}

#[derive(Debug, Clone)]
pub enum HoverAction {
    Runnable(Runnable),
    Implementaion(FilePosition),
    GoToType(Vec<HoverGotoTypeData>),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct HoverGotoTypeData {
    pub mod_path: String,
    pub nav: NavigationTarget,
}

/// Contains the results when hovering over an item
#[derive(Debug, Default)]
pub struct HoverResult {
    results: Vec<String>,
    actions: Vec<HoverAction>,
}

impl HoverResult {
    pub fn new() -> HoverResult {
        Self::default()
    }

    pub fn extend(&mut self, item: Option<String>) {
        self.results.extend(item);
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    pub fn len(&self) -> usize {
        self.results.len()
    }

    pub fn first(&self) -> Option<&str> {
        self.results.first().map(String::as_str)
    }

    pub fn results(&self) -> &[String] {
        &self.results
    }

    pub fn actions(&self) -> &[HoverAction] {
        &self.actions
    }

    pub fn push_action(&mut self, action: HoverAction) {
        self.actions.push(action);
    }

    /// Returns the results converted into markup
    /// for displaying in a UI
    ///
    /// Does not process actions!
    pub fn to_markup(&self) -> String {
        self.results.join("\n\n___\n")
    }
}

// Feature: Hover
//
// Shows additional information, like type of an expression or documentation for definition when "focusing" code.
// Focusing is usually hovering with a mouse, but can also be triggered with a shortcut.
pub(crate) fn hover(db: &RootDatabase, position: FilePosition) -> Option<RangeInfo<HoverResult>> {
    let sema = Semantics::new(db);
    let file = sema.parse(position.file_id).syntax().clone();
    let token = pick_best(file.token_at_offset(position.offset))?;
    let token = sema.descend_into_macros(token);

    let mut res = HoverResult::new();

    if let Some((node, name_kind)) = match_ast! {
        match (token.parent()) {
            ast::NameRef(name_ref) => {
                classify_name_ref(&sema, &name_ref).map(|d| (name_ref.syntax().clone(), d.definition()))
            },
            ast::Name(name) => {
                classify_name(&sema, &name).map(|d| (name.syntax().clone(), d.definition()))
            },
            _ => None,
        }
    } {
        let range = sema.original_range(&node).range;
        let text = hover_text_from_name_kind(db, name_kind.clone());
        let text = text.map(|text| rewrite_links(db, &text, &name_kind).unwrap_or(text));
        res.extend(text);

        if !res.is_empty() {
            if let Some(action) = show_implementations_action(db, name_kind) {
                res.push_action(action);
            }

            if let Some(action) = runnable_action(&sema, name_kind, position.file_id) {
                res.push_action(action);
            }

            if let Some(action) = goto_type_action(db, name_kind) {
                res.push_action(action);
            }

            return Some(RangeInfo::new(range, res));
        }
    }

    let node = token
        .ancestors()
        .find(|n| ast::Expr::cast(n.clone()).is_some() || ast::Pat::cast(n.clone()).is_some())?;

    let ty = match_ast! {
        match node {
            ast::MacroCall(_it) => {
                // If this node is a MACRO_CALL, it means that `descend_into_macros` failed to resolve.
                // (e.g expanding a builtin macro). So we give up here.
                return None;
            },
            ast::Expr(it) => {
                sema.type_of_expr(&it)
            },
            ast::Pat(it) => {
                sema.type_of_pat(&it)
            },
            _ => None,
        }
    }?;

    res.extend(Some(rust_code_markup(&ty.display(db))));
    let range = sema.original_range(&node).range;
    Some(RangeInfo::new(range, res))
}

fn show_implementations_action(db: &RootDatabase, def: Definition) -> Option<HoverAction> {
    fn to_action(nav_target: NavigationTarget) -> HoverAction {
        HoverAction::Implementaion(FilePosition {
            file_id: nav_target.file_id(),
            offset: nav_target.range().start(),
        })
    }

    match def {
        Definition::ModuleDef(it) => match it {
            ModuleDef::Adt(Adt::Struct(it)) => Some(to_action(it.to_nav(db))),
            ModuleDef::Adt(Adt::Union(it)) => Some(to_action(it.to_nav(db))),
            ModuleDef::Adt(Adt::Enum(it)) => Some(to_action(it.to_nav(db))),
            ModuleDef::Trait(it) => Some(to_action(it.to_nav(db))),
            _ => None,
        },
        _ => None,
    }
}

fn runnable_action(
    sema: &Semantics<RootDatabase>,
    def: Definition,
    file_id: FileId,
) -> Option<HoverAction> {
    match def {
        Definition::ModuleDef(it) => match it {
            ModuleDef::Module(it) => match it.definition_source(sema.db).value {
                ModuleSource::Module(it) => runnable(&sema, it.syntax().clone(), file_id)
                    .map(|it| HoverAction::Runnable(it)),
                _ => None,
            },
            ModuleDef::Function(it) => {
                let src = it.source(sema.db);
                if src.file_id != file_id.into() {
                    mark::hit!(hover_macro_generated_struct_fn_doc_comment);
                    mark::hit!(hover_macro_generated_struct_fn_doc_attr);

                    return None;
                }

                runnable(&sema, src.value.syntax().clone(), file_id)
                    .map(|it| HoverAction::Runnable(it))
            }
            _ => None,
        },
        _ => None,
    }
}

fn goto_type_action(db: &RootDatabase, def: Definition) -> Option<HoverAction> {
    match def {
        Definition::Local(it) => {
            let mut targets: Vec<ModuleDef> = Vec::new();
            let mut push_new_def = |item: ModuleDef| {
                if !targets.contains(&item) {
                    targets.push(item);
                }
            };

            it.ty(db).walk(db, |t| {
                if let Some(adt) = t.as_adt() {
                    push_new_def(adt.into());
                } else if let Some(trait_) = t.as_dyn_trait() {
                    push_new_def(trait_.into());
                } else if let Some(traits) = t.as_impl_traits(db) {
                    traits.into_iter().for_each(|it| push_new_def(it.into()));
                } else if let Some(trait_) = t.as_associated_type_parent_trait(db) {
                    push_new_def(trait_.into());
                }
            });

            let targets = targets
                .into_iter()
                .filter_map(|it| {
                    Some(HoverGotoTypeData {
                        mod_path: mod_path(db, &it)?,
                        nav: it.try_to_nav(db)?,
                    })
                })
                .collect();

            Some(HoverAction::GoToType(targets))
        }
        _ => None,
    }
}

fn hover_text(
    docs: Option<String>,
    desc: Option<String>,
    mod_path: Option<String>,
) -> Option<String> {
    if let Some(desc) = desc {
        Some(rust_code_markup_with_doc(&desc, docs.as_deref(), mod_path.as_deref()))
    } else {
        docs
    }
}

fn definition_owner_name(db: &RootDatabase, def: &Definition) -> Option<String> {
    match def {
        Definition::Field(f) => Some(f.parent_def(db).name(db)),
        Definition::Local(l) => l.parent(db).name(db),
        Definition::ModuleDef(md) => match md {
            ModuleDef::Function(f) => match f.as_assoc_item(db)?.container(db) {
                AssocItemContainer::Trait(t) => Some(t.name(db)),
                AssocItemContainer::ImplDef(i) => i.target_ty(db).as_adt().map(|adt| adt.name(db)),
            },
            ModuleDef::EnumVariant(e) => Some(e.parent_enum(db).name(db)),
            _ => None,
        },
        Definition::SelfType(i) => i.target_ty(db).as_adt().map(|adt| adt.name(db)),
        _ => None,
    }
    .map(|name| name.to_string())
}

fn determine_mod_path(db: &RootDatabase, module: Module, name: Option<String>) -> String {
    once(db.crate_graph()[module.krate().into()].display_name.as_ref().map(ToString::to_string))
        .chain(
            module
                .path_to_root(db)
                .into_iter()
                .rev()
                .map(|it| it.name(db).map(|name| name.to_string())),
        )
        .chain(once(name))
        .flatten()
        .join("::")
}

// returns None only for ModuleDef::BuiltinType
fn mod_path(db: &RootDatabase, item: &ModuleDef) -> Option<String> {
    Some(determine_mod_path(db, item.module(db)?, item.name(db).map(|name| name.to_string())))
}

fn definition_mod_path(db: &RootDatabase, def: &Definition) -> Option<String> {
    def.module(db).map(|module| determine_mod_path(db, module, definition_owner_name(db, def)))
}

fn hover_text_from_name_kind(db: &RootDatabase, def: Definition) -> Option<String> {
    let mod_path = definition_mod_path(db, &def);
    return match def {
        Definition::Macro(it) => {
            let src = it.source(db);
            let docs = Documentation::from_ast(&src.value).map(Into::into);
            hover_text(docs, Some(macro_label(&src.value)), mod_path)
        }
        Definition::Field(it) => {
            let src = it.source(db);
            match src.value {
                FieldSource::Named(it) => {
                    let docs = Documentation::from_ast(&it).map(Into::into);
                    hover_text(docs, it.short_label(), mod_path)
                }
                _ => None,
            }
        }
        Definition::ModuleDef(it) => match it {
            ModuleDef::Module(it) => match it.definition_source(db).value {
                ModuleSource::Module(it) => {
                    let docs = Documentation::from_ast(&it).map(Into::into);
                    hover_text(docs, it.short_label(), mod_path)
                }
                _ => None,
            },
            ModuleDef::Function(it) => from_def_source(db, it, mod_path),
            ModuleDef::Adt(Adt::Struct(it)) => from_def_source(db, it, mod_path),
            ModuleDef::Adt(Adt::Union(it)) => from_def_source(db, it, mod_path),
            ModuleDef::Adt(Adt::Enum(it)) => from_def_source(db, it, mod_path),
            ModuleDef::EnumVariant(it) => from_def_source(db, it, mod_path),
            ModuleDef::Const(it) => from_def_source(db, it, mod_path),
            ModuleDef::Static(it) => from_def_source(db, it, mod_path),
            ModuleDef::Trait(it) => from_def_source(db, it, mod_path),
            ModuleDef::TypeAlias(it) => from_def_source(db, it, mod_path),
            ModuleDef::BuiltinType(it) => Some(it.to_string()),
        },
        Definition::Local(it) => Some(rust_code_markup(&it.ty(db).display(db))),
        Definition::TypeParam(_) | Definition::SelfType(_) => {
            // FIXME: Hover for generic param
            None
        }
    };

    fn from_def_source<A, D>(db: &RootDatabase, def: D, mod_path: Option<String>) -> Option<String>
    where
        D: HasSource<Ast = A>,
        A: ast::DocCommentsOwner + ast::NameOwner + ShortLabel + ast::AttrsOwner,
    {
        let src = def.source(db);
        let docs = Documentation::from_ast(&src.value).map(Into::into);
        hover_text(docs, src.value.short_label(), mod_path)
    }
}

// Rewrites a markdown document, resolving links using `callback` and additionally striping prefixes/suffixes on link titles.
fn map_links<'e>(
    events: impl Iterator<Item = Event<'e>>,
    callback: impl Fn(&str, &str) -> String,
) -> impl Iterator<Item = Event<'e>> {
    let mut in_link = false;
    let mut link_text = CowStr::Borrowed("");
    events.map(move |evt| match evt {
        Event::Start(Tag::Link(..)) => {
            in_link = true;
            evt
        }
        Event::End(Tag::Link(link_type, target, _)) => {
            in_link = false;
            let target = callback(&target, &link_text);
            Event::End(Tag::Link(link_type, CowStr::Boxed(target.into()), CowStr::Borrowed("")))
        }
        Event::Text(s) if in_link => {
            link_text = s.clone();
            Event::Text(CowStr::Boxed(strip_prefixes_suffixes(&s).into()))
        }
        Event::Code(s) if in_link => {
            link_text = s.clone();
            Event::Code(CowStr::Boxed(strip_prefixes_suffixes(&s).into()))
        }
        _ => evt,
    })
}

/// Rewrite documentation links in markdown to point to local documentation/docs.rs
fn rewrite_links(db: &RootDatabase, markdown: &str, definition: &Definition) -> Option<String> {
    let doc = Parser::new_with_broken_link_callback(
        markdown,
        Options::empty(),
        Some(&|label, _| Some((/*url*/ label.to_string(), /*title*/ label.to_string()))),
    );

    let doc = map_links(doc, |target, title: &str| {
        match Url::parse(target) {
            // If this is a valid absolute URL don't touch it
            Ok(_) => target.to_string(),
            // Otherwise there are two main possibilities
            // path-based links: `../../module/struct.MyStruct.html`
            // module-based links (AKA intra-doc links): `super::super::module::MyStruct`
            Err(_) => {
                let resolved = try_resolve_intra(db, definition, title, &target)
                    .or_else(|| try_resolve_path(db, definition, &target));

                if let Some(resolved) = resolved {
                    resolved
                } else {
                    target.to_string()
                }
            }
        }
    });
    let mut out = String::new();
    cmark(doc, &mut out, None).ok();
    Some(out)
}

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
enum Namespace {
    Types,
    Values,
    Macros,
}

lazy_static!(
    /// Map of namespaces to identifying prefixes and suffixes as defined by RFC1946.
    static ref NS_MAP: HashMap<Namespace, (HashSet<&'static str>, HashSet<&'static str>)> = hashmap!{
        Namespace::Types => (hashset!{"type", "struct", "enum", "mod", "trait", "union", "module"}, hashset!{}),
        Namespace::Values => (hashset!{"value", "function", "fn", "method", "const", "static", "mod", "module"}, hashset!{"()"}),
        Namespace::Macros => (hashset!{"macro"}, hashset!{"!"})
    };
);

impl Namespace {
    /// Extract the specified namespace from an intra-doc-link if one exists.
    fn from_intra_spec(s: &str) -> Option<Self> {
        NS_MAP
            .iter()
            .filter(|(_ns, (prefixes, suffixes))| {
                prefixes
                    .iter()
                    .map(|prefix| {
                        s.starts_with(prefix)
                            && s.chars()
                                .nth(prefix.len() + 1)
                                .map(|c| c == '@' || c == ' ')
                                .unwrap_or(false)
                    })
                    .any(|cond| cond)
                    || suffixes
                        .iter()
                        .map(|suffix| {
                            s.starts_with(suffix)
                                && s.chars()
                                    .nth(suffix.len() + 1)
                                    .map(|c| c == '@' || c == ' ')
                                    .unwrap_or(false)
                        })
                        .any(|cond| cond)
            })
            .map(|(ns, (_, _))| *ns)
            .next()
    }
}

// Strip prefixes, suffixes, and inline code marks from the given string.
fn strip_prefixes_suffixes(mut s: &str) -> &str {
    s = s.trim_matches('`');
    NS_MAP.iter().for_each(|(_, (prefixes, suffixes))| {
        prefixes.iter().for_each(|prefix| s = s.trim_start_matches(prefix));
        suffixes.iter().for_each(|suffix| s = s.trim_end_matches(suffix));
    });
    s.trim_start_matches("@").trim()
}

/// Try to resolve path to local documentation via intra-doc-links (i.e. `super::gateway::Shard`).
///
/// See [RFC1946](https://github.com/rust-lang/rfcs/blob/master/text/1946-intra-rustdoc-links.md).
fn try_resolve_intra(
    db: &RootDatabase,
    definition: &Definition,
    link_text: &str,
    link_target: &str,
) -> Option<String> {
    // Set link_target for implied shortlinks
    let link_target =
        if link_target.is_empty() { link_text.trim_matches('`') } else { link_target };

    // Namespace disambiguation
    let namespace = Namespace::from_intra_spec(link_target);

    // Strip prefixes/suffixes
    let link_target = strip_prefixes_suffixes(link_target);

    // Parse link as a module path
    // This expects a full document, which a single path isn't, but we can just ignore the errors.
    let parsed = SyntaxNode::new_root(ra_syntax::parse_text(link_target).0);
    let path = parsed.descendants().filter_map(Path::cast).next()?;
    let modpath = ModPath::from_src(path, &Hygiene::new_unhygienic()).unwrap();

    // Resolve it relative to symbol's location (according to the RFC this should consider small scopes
    let resolver = definition.resolver(db)?;

    let resolved = resolver.resolve_module_path_in_items(db, &modpath);
    let (defid, namespace) = match namespace {
        // FIXME: .or(resolved.macros)
        None => resolved
            .types
            .map(|t| (t.0, Namespace::Types))
            .or(resolved.values.map(|t| (t.0, Namespace::Values)))?,
        Some(ns @ Namespace::Types) => (resolved.types?.0, ns),
        Some(ns @ Namespace::Values) => (resolved.values?.0, ns),
        // FIXME:
        Some(Namespace::Macros) => None?,
    };

    // Get the filepath of the final symbol
    let def: ModuleDef = defid.into();
    let module = def.module(db)?;
    let krate = module.krate();
    let ns = match namespace {
        Namespace::Types => ItemInNs::Types(defid),
        Namespace::Values => ItemInNs::Values(defid),
        // FIXME:
        Namespace::Macros => None?,
    };
    let import_map = db.import_map(krate.into());
    let path = import_map.path_of(ns)?;

    Some(
        get_doc_url(db, &krate)?
            .join(&format!("{}/", krate.display_name(db)?))
            .ok()?
            .join(&path.segments.iter().map(|name| format!("{}", name)).join("/"))
            .ok()?
            .join(&get_symbol_filename(db, &Definition::ModuleDef(def))?)
            .ok()?
            .into_string(),
    )
}

/// Try to resolve path to local documentation via path-based links (i.e. `../gateway/struct.Shard.html`).
fn try_resolve_path(db: &RootDatabase, definition: &Definition, link: &str) -> Option<String> {
    if !link.contains("#") && !link.contains(".html") {
        return None;
    }
    let ns = if let Definition::ModuleDef(moddef) = definition {
        ItemInNs::Types(moddef.clone().into())
    } else {
        return None;
    };
    let module = definition.module(db)?;
    let krate = module.krate();
    let import_map = db.import_map(krate.into());
    let base = once(format!("{}", krate.display_name(db)?))
        .chain(import_map.path_of(ns)?.segments.iter().map(|name| format!("{}", name)))
        .join("/");

    get_doc_url(db, &krate)
        .and_then(|url| url.join(&base).ok())
        .and_then(|url| {
            get_symbol_filename(db, definition).as_deref().map(|f| url.join(f).ok()).flatten()
        })
        .and_then(|url| url.join(link).ok())
        .map(|url| url.into_string())
}

/// Try to get the root URL of the documentation of a crate.
// FIXME: Special case standard, core, alloc libraries
fn get_doc_url(db: &RootDatabase, krate: &Crate) -> Option<Url> {
    // Look for #![doc(html_root_url = "...")]
    let attrs = db.attrs(AttrDef::from(krate.root_module(db)?).into());
    let doc_attr_q = attrs.by_key("doc");
    let doc_url = if doc_attr_q.exists() {
        doc_attr_q.tt_values().filter_map(|tt| match tt.token_trees.as_slice() {
            &[
                TokenTree::Leaf(Leaf::Ident(Ident{text: ref ident_text, ..})),
                TokenTree::Leaf(Leaf::Punct(Punct{r#char: '=', ..})),
                TokenTree::Leaf(Leaf::Literal(Literal{ref text, ..}))
            ] if ident_text == "html_root_url" => Some(text.to_string()),
            _ => {
                None
            }
        }).next()
    } else {
        // Fallback to docs.rs
        // FIXME: Specify an exact version here (from Cargo.lock)
        Some(format!("https://docs.rs/{}/*", krate.display_name(db)?))
    };

    doc_url.map(|s| s.trim_matches('"').to_owned() + "/").and_then(|s| Url::parse(&s).ok())
}

/// Get the filename and extension generated for a symbol by rustdoc.
///
/// Example: `struct.Shard.html`
fn get_symbol_filename(db: &RootDatabase, definition: &Definition) -> Option<String> {
    Some(match definition {
        Definition::ModuleDef(def) => match def {
            ModuleDef::Adt(adt) => match adt {
                Adt::Struct(s) => format!("struct.{}.html", s.name(db)),
                Adt::Enum(e) => format!("enum.{}.html", e.name(db)),
                Adt::Union(u) => format!("union.{}.html", u.name(db)),
            },
            ModuleDef::Module(_) => "index.html".to_string(),
            ModuleDef::Trait(t) => format!("trait.{}.html", t.name(db)),
            ModuleDef::TypeAlias(t) => format!("type.{}.html", t.name(db)),
            ModuleDef::BuiltinType(t) => format!("primitive.{}.html", t.as_name()),
            ModuleDef::Function(f) => format!("fn.{}.html", f.name(db)),
            ModuleDef::EnumVariant(ev) => {
                format!("enum.{}.html#variant.{}", ev.parent_enum(db).name(db), ev.name(db))
            }
            ModuleDef::Const(c) => format!("const.{}.html", c.name(db)?),
            ModuleDef::Static(s) => format!("static.{}.html", s.name(db)?),
        },
        Definition::Macro(m) => format!("macro.{}.html", m.name(db)?),
        _ => None?,
    })
}

fn pick_best(tokens: TokenAtOffset<SyntaxToken>) -> Option<SyntaxToken> {
    return tokens.max_by_key(priority);
    fn priority(n: &SyntaxToken) -> usize {
        match n.kind() {
            IDENT | INT_NUMBER => 3,
            L_PAREN | R_PAREN => 2,
            kind if kind.is_trivia() => 0,
            _ => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_debug_snapshot;

    use ra_db::FileLoader;
    use ra_syntax::TextRange;

    use crate::mock_analysis::analysis_and_position;

    fn trim_markup(s: &str) -> String {
        s.trim()
            .replace("````", "```")
            .replace("---", "___")
            .replace("\\<-", "<-")
            .replace("```\n\n___", "```\n___")
            .trim_start_matches("```rust\n")
            .trim_start_matches("test\n```\n\n```rust\n")
            .trim_end_matches("\n```")
            .to_string()
    }

    fn trim_markup_opt(s: Option<&str>) -> Option<String> {
        s.map(trim_markup)
    }

    fn assert_impl_action(action: &HoverAction, position: u32) {
        let offset = match action {
            HoverAction::Implementaion(pos) => pos.offset,
            it => panic!("Unexpected hover action: {:#?}", it),
        };
        assert_eq!(offset, position.into());
    }

    fn check_hover_result(fixture: &str, expected: &[&str]) -> (String, Vec<HoverAction>) {
        let (analysis, position) = analysis_and_position(fixture);
        let hover = analysis.hover(position).unwrap().unwrap();
        let mut results = Vec::from(hover.info.results());
        results.sort();

        for (markup, expected) in
            results.iter().zip(expected.iter().chain(std::iter::repeat(&"<missing>")))
        {
            assert_eq!(trim_markup(&markup), *expected);
        }

        assert_eq!(hover.info.len(), expected.len());

        let content = analysis.db.file_text(position.file_id);
        (content[hover.range].to_string(), hover.info.actions().to_vec())
    }

    fn check_hover_no_result(fixture: &str) {
        let (analysis, position) = analysis_and_position(fixture);
        assert!(analysis.hover(position).unwrap().is_none());
    }

    #[test]
    fn hover_shows_type_of_an_expression() {
        let (analysis, position) = analysis_and_position(
            r#"
pub fn foo() -> u32 { 1 }

fn main() {
    let foo_test = foo()<|>;
}
"#,
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(hover.range, TextRange::new(58.into(), 63.into()));
        assert_eq!(trim_markup_opt(hover.info.first()).as_deref(), Some("u32"));
    }

    #[test]
    fn hover_shows_long_type_of_an_expression() {
        check_hover_result(
            r#"
            //- /main.rs
            struct Scan<A, B, C> {
                a: A,
                b: B,
                c: C,
            }

            struct FakeIter<I> {
                inner: I,
            }

            struct OtherStruct<T> {
                i: T,
            }

            enum FakeOption<T> {
                Some(T),
                None,
            }

            fn scan<A, B, C>(a: A, b: B, c: C) -> FakeIter<Scan<OtherStruct<A>, B, C>> {
                FakeIter { inner: Scan { a, b, c } }
            }

            fn main() {
                let num: i32 = 55;
                let closure = |memo: &mut u32, value: &u32, _another: &mut u32| -> FakeOption<u32> {
                    FakeOption::Some(*memo + value)
                };
                let number = 5u32;
                let mut iter<|> = scan(OtherStruct { i: num }, closure, number);
            }
            "#,
            &["FakeIter<Scan<OtherStruct<OtherStruct<i32>>, |&mut u32, &u32, &mut u32| -> FakeOption<u32>, u32>>"],
        );
    }

    #[test]
    fn hover_shows_fn_signature() {
        // Single file with result
        check_hover_result(
            r#"
            //- /main.rs
            pub fn foo() -> u32 { 1 }

            fn main() {
                let foo_test = fo<|>o();
            }
        "#,
            &["pub fn foo() -> u32"],
        );

        // Multiple candidates but results are ambiguous.
        check_hover_result(
            r#"
            //- /a.rs
            pub fn foo() -> u32 { 1 }

            //- /b.rs
            pub fn foo() -> &str { "" }

            //- /c.rs
            pub fn foo(a: u32, b: u32) {}

            //- /main.rs
            mod a;
            mod b;
            mod c;

            fn main() {
                let foo_test = fo<|>o();
            }
        "#,
            &["{unknown}"],
        );
    }

    #[test]
    fn hover_shows_fn_signature_with_type_params() {
        check_hover_result(
            r#"
            //- /main.rs
            pub fn foo<'a, T: AsRef<str>>(b: &'a T) -> &'a str { }

            fn main() {
                let foo_test = fo<|>o();
            }
        "#,
            &["pub fn foo<'a, T: AsRef<str>>(b: &'a T) -> &'a str"],
        );
    }

    #[test]
    fn hover_shows_fn_signature_on_fn_name() {
        check_hover_result(
            r#"
            //- /main.rs
            pub fn foo<|>(a: u32, b: u32) -> u32 {}

            fn main() {
            }
        "#,
            &["pub fn foo(a: u32, b: u32) -> u32"],
        );
    }

    #[test]
    fn hover_shows_struct_field_info() {
        // Hovering over the field when instantiating
        check_hover_result(
            r#"
            //- /main.rs
            struct Foo {
                field_a: u32,
            }

            fn main() {
                let foo = Foo {
                    field_a<|>: 0,
                };
            }
        "#,
            &["test::Foo\n```\n\n```rust\nfield_a: u32"],
        );

        // Hovering over the field in the definition
        check_hover_result(
            r#"
            //- /main.rs
            struct Foo {
                field_a<|>: u32,
            }

            fn main() {
                let foo = Foo {
                    field_a: 0,
                };
            }
        "#,
            &["test::Foo\n```\n\n```rust\nfield_a: u32"],
        );
    }

    #[test]
    fn hover_const_static() {
        check_hover_result(
            r#"
            //- /main.rs
            const foo<|>: u32 = 0;
        "#,
            &["const foo: u32"],
        );

        check_hover_result(
            r#"
            //- /main.rs
            static foo<|>: u32 = 0;
        "#,
            &["static foo: u32"],
        );
    }

    #[test]
    fn hover_default_generic_types() {
        check_hover_result(
            r#"
//- /main.rs
struct Test<K, T = u8> {
    k: K,
    t: T,
}

fn main() {
    let zz<|> = Test { t: 23u8, k: 33 };
}"#,
            &["Test<i32, u8>"],
        );
    }

    #[test]
    fn hover_some() {
        let (analysis, position) = analysis_and_position(
            "
            enum Option<T> { Some(T) }
            use Option::Some;

            fn main() {
                So<|>me(12);
            }
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(
            trim_markup_opt(hover.info.first()).as_deref(),
            Some("test::Option\n```\n\n```rust\nSome")
        );

        let (analysis, position) = analysis_and_position(
            "
            enum Option<T> { Some(T) }
            use Option::Some;

            fn main() {
                let b<|>ar = Some(12);
            }
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()).as_deref(), Some("Option<i32>"));
    }

    #[test]
    fn hover_enum_variant() {
        check_hover_result(
            r#"
            //- /main.rs
            enum Option<T> {
                /// The None variant
                Non<|>e
            }
        "#,
            &["
test::Option
```

```rust
None
```
___

The None variant
            "
            .trim()],
        );

        check_hover_result(
            r#"
            //- /main.rs
            enum Option<T> {
                /// The Some variant
                Some(T)
            }
            fn main() {
                let s = Option::Som<|>e(12);
            }
        "#,
            &["
test::Option
```

```rust
Some
```
___

The Some variant
            "
            .trim()],
        );
    }

    #[test]
    fn hover_for_local_variable() {
        let (analysis, position) = analysis_and_position("fn func(foo: i32) { fo<|>o; }");
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()).as_deref(), Some("i32"));
    }

    #[test]
    fn hover_for_local_variable_pat() {
        let (analysis, position) = analysis_and_position("fn func(fo<|>o: i32) {}");
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()).as_deref(), Some("i32"));
    }

    #[test]
    fn hover_local_var_edge() {
        let (analysis, position) = analysis_and_position(
            "
fn func(foo: i32) { if true { <|>foo; }; }
",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()).as_deref(), Some("i32"));
    }

    #[test]
    fn hover_for_param_edge() {
        let (analysis, position) = analysis_and_position("fn func(<|>foo: i32) {}");
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()).as_deref(), Some("i32"));
    }

    #[test]
    fn test_hover_infer_associated_method_result() {
        let (analysis, position) = analysis_and_position(
            "
            struct Thing { x: u32 }

            impl Thing {
                fn new() -> Thing {
                    Thing { x: 0 }
                }
            }

            fn main() {
                let foo_<|>test = Thing::new();
            }
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()).as_deref(), Some("Thing"));
    }

    #[test]
    fn test_hover_infer_associated_method_exact() {
        let (analysis, position) = analysis_and_position(
            "
            mod wrapper {
                struct Thing { x: u32 }

                impl Thing {
                    fn new() -> Thing {
                        Thing { x: 0 }
                    }
                }
            }

            fn main() {
                let foo_test = wrapper::Thing::new<|>();
            }
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(
            trim_markup_opt(hover.info.first()).as_deref(),
            Some("test::wrapper::Thing\n```\n\n```rust\nfn new() -> Thing")
        );
    }

    #[test]
    fn test_hover_infer_associated_const_in_pattern() {
        let (analysis, position) = analysis_and_position(
            "
            struct X;
            impl X {
                const C: u32 = 1;
            }

            fn main() {
                match 1 {
                    X::C<|> => {},
                    2 => {},
                    _ => {}
                };
            }
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()).as_deref(), Some("const C: u32"));
    }

    #[test]
    fn test_hover_self() {
        let (analysis, position) = analysis_and_position(
            "
            struct Thing { x: u32 }
            impl Thing {
                fn new() -> Self {
                    Self<|> { x: 0 }
                }
            }
        ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()).as_deref(), Some("Thing"));

        /* FIXME: revive these tests
                let (analysis, position) = analysis_and_position(
                    "
                    struct Thing { x: u32 }
                    impl Thing {
                        fn new() -> Self<|> {
                            Self { x: 0 }
                        }
                    }
                    ",
                );

                let hover = analysis.hover(position).unwrap().unwrap();
                assert_eq!(trim_markup_opt(hover.info.first()), Some("Thing"));

                let (analysis, position) = analysis_and_position(
                    "
                    enum Thing { A }
                    impl Thing {
                        pub fn new() -> Self<|> {
                            Thing::A
                        }
                    }
                    ",
                );
                let hover = analysis.hover(position).unwrap().unwrap();
                assert_eq!(trim_markup_opt(hover.info.first()), Some("enum Thing"));

                let (analysis, position) = analysis_and_position(
                    "
                    enum Thing { A }
                    impl Thing {
                        pub fn thing(a: Self<|>) {
                        }
                    }
                    ",
                );
                let hover = analysis.hover(position).unwrap().unwrap();
                assert_eq!(trim_markup_opt(hover.info.first()), Some("enum Thing"));
        */
    }

    #[test]
    fn test_hover_shadowing_pat() {
        let (analysis, position) = analysis_and_position(
            "
            fn x() {}

            fn y() {
                let x = 0i32;
                x<|>;
            }
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()).as_deref(), Some("i32"));
    }

    #[test]
    fn test_hover_macro_invocation() {
        let (analysis, position) = analysis_and_position(
            "
            macro_rules! foo {
                () => {}
            }

            fn f() {
                fo<|>o!();
            }
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()).as_deref(), Some("macro_rules! foo"));
    }

    #[test]
    fn test_hover_tuple_field() {
        let (analysis, position) = analysis_and_position(
            "
            struct TS(String, i32<|>);
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()).as_deref(), Some("i32"));
    }

    #[test]
    fn test_hover_through_macro() {
        let (hover_on, _) = check_hover_result(
            "
            //- /lib.rs
            macro_rules! id {
                ($($tt:tt)*) => { $($tt)* }
            }
            fn foo() {}
            id! {
                fn bar() {
                    fo<|>o();
                }
            }
            ",
            &["fn foo()"],
        );

        assert_eq!(hover_on, "foo")
    }

    #[test]
    fn test_hover_through_expr_in_macro() {
        let (hover_on, _) = check_hover_result(
            "
            //- /lib.rs
            macro_rules! id {
                ($($tt:tt)*) => { $($tt)* }
            }
            fn foo(bar:u32) {
                let a = id!(ba<|>r);
            }
            ",
            &["u32"],
        );

        assert_eq!(hover_on, "bar")
    }

    #[test]
    fn test_hover_through_expr_in_macro_recursive() {
        let (hover_on, _) = check_hover_result(
            "
            //- /lib.rs
            macro_rules! id_deep {
                ($($tt:tt)*) => { $($tt)* }
            }
            macro_rules! id {
                ($($tt:tt)*) => { id_deep!($($tt)*) }
            }
            fn foo(bar:u32) {
                let a = id!(ba<|>r);
            }
            ",
            &["u32"],
        );

        assert_eq!(hover_on, "bar")
    }

    #[test]
    fn test_hover_through_func_in_macro_recursive() {
        let (hover_on, _) = check_hover_result(
            "
            //- /lib.rs
            macro_rules! id_deep {
                ($($tt:tt)*) => { $($tt)* }
            }
            macro_rules! id {
                ($($tt:tt)*) => { id_deep!($($tt)*) }
            }
            fn bar() -> u32 {
                0
            }
            fn foo() {
                let a = id!([0u32, bar(<|>)] );
            }
            ",
            &["u32"],
        );

        assert_eq!(hover_on, "bar()")
    }

    #[test]
    fn test_hover_through_literal_string_in_macro() {
        let (hover_on, _) = check_hover_result(
            r#"
            //- /lib.rs
            macro_rules! arr {
                ($($tt:tt)*) => { [$($tt)*)] }
            }
            fn foo() {
                let mastered_for_itunes = "";
                let _ = arr!("Tr<|>acks", &mastered_for_itunes);
            }
            "#,
            &["&str"],
        );

        assert_eq!(hover_on, "\"Tracks\"");
    }

    #[test]
    fn test_hover_through_assert_macro() {
        let (hover_on, _) = check_hover_result(
            r#"
            //- /lib.rs
            #[rustc_builtin_macro]
            macro_rules! assert {}

            fn bar() -> bool { true }
            fn foo() {
                assert!(ba<|>r());
            }
            "#,
            &["fn bar() -> bool"],
        );

        assert_eq!(hover_on, "bar");
    }

    #[test]
    fn test_hover_through_literal_string_in_builtin_macro() {
        check_hover_no_result(
            r#"
            //- /lib.rs
            #[rustc_builtin_macro]
            macro_rules! format {}

            fn foo() {
                format!("hel<|>lo {}", 0);
            }
            "#,
        );
    }

    #[test]
    fn test_hover_non_ascii_space_doc() {
        check_hover_result(
            "
            //- /lib.rs
            ///　<- `\u{3000}` here
            fn foo() {
            }

            fn bar() {
                fo<|>o();
            }
            ",
            &["fn foo()\n```\n___\n\n<- `\u{3000}` here"],
        );
    }

    #[test]
    fn test_hover_function_show_qualifiers() {
        check_hover_result(
            "
            //- /lib.rs
            async fn foo<|>() {}
            ",
            &["async fn foo()"],
        );
        check_hover_result(
            "
            //- /lib.rs
            pub const unsafe fn foo<|>() {}
            ",
            &["pub const unsafe fn foo()"],
        );
        check_hover_result(
            r#"
            //- /lib.rs
            pub(crate) async unsafe extern "C" fn foo<|>() {}
            "#,
            &[r#"pub(crate) async unsafe extern "C" fn foo()"#],
        );
    }

    #[test]
    fn test_hover_trait_show_qualifiers() {
        let (_, actions) = check_hover_result(
            "
            //- /lib.rs
            unsafe trait foo<|>() {}
            ",
            &["unsafe trait foo"],
        );
        assert_impl_action(&actions[0], 13);
    }

    #[test]
    fn test_hover_mod_with_same_name_as_function() {
        check_hover_result(
            "
            //- /lib.rs
            use self::m<|>y::Bar;

            mod my {
                pub struct Bar;
            }

            fn my() {}
            ",
            &["mod my"],
        );
    }

    #[test]
    fn test_hover_struct_doc_comment() {
        check_hover_result(
            r#"
            //- /lib.rs
            /// bar docs
            struct Bar;

            fn foo() {
                let bar = Ba<|>r;
            }
            "#,
            &["struct Bar\n```\n___\n\nbar docs"],
        );
    }

    #[test]
    fn test_hover_struct_doc_attr() {
        check_hover_result(
            r#"
            //- /lib.rs
            #[doc = "bar docs"]
            struct Bar;

            fn foo() {
                let bar = Ba<|>r;
            }
            "#,
            &["struct Bar\n```\n___\n\nbar docs"],
        );
    }

    #[test]
    fn test_hover_struct_doc_attr_multiple_and_mixed() {
        check_hover_result(
            r#"
            //- /lib.rs
            /// bar docs 0
            #[doc = "bar docs 1"]
            #[doc = "bar docs 2"]
            struct Bar;

            fn foo() {
                let bar = Ba<|>r;
            }
            "#,
            &["struct Bar\n```\n___\n\nbar docs 0\n\nbar docs 1\n\nbar docs 2"],
        );
    }

    #[test]
    fn test_hover_path_link() {
        check_hover_result(
            r"
            //- /lib.rs
            pub struct Foo;
            /// [Foo](struct.Foo.html)
            pub struct B<|>ar
            ",
            &["pub struct Bar\n```\n___\n\n[Foo](https://docs.rs/test/*/test/struct.Foo.html)"],
        );
    }

    #[test]
    fn test_hover_intra_link() {
        check_hover_result(
            r"
            //- /lib.rs
            pub struct Foo;
            /// [Foo](Foo)
            pub struct B<|>ar
            ",
            &["pub struct Bar\n```\n___\n\n[Foo](https://docs.rs/test/*/test/struct.Foo.html)"],
        );
    }

    #[test]
    fn test_hover_intra_link_shortlink() {
        check_hover_result(
            r"
            //- /lib.rs
            pub struct Foo;
            /// [Foo]
            pub struct B<|>ar
            ",
            &["pub struct Bar\n```\n___\n\n[Foo](https://docs.rs/test/*/test/struct.Foo.html)"],
        );
    }

    #[test]
    fn test_hover_intra_link_shortlink_code() {
        check_hover_result(
            r"
            //- /lib.rs
            pub struct Foo;
            /// [`Foo`]
            pub struct B<|>ar
            ",
            &["pub struct Bar\n```\n___\n\n[`Foo`](https://docs.rs/test/*/test/struct.Foo.html)"],
        );
    }

    #[test]
    fn test_hover_intra_link_namespaced() {
        check_hover_result(
            r"
            //- /lib.rs
            pub struct Foo;
            fn Foo() {}
            /// [Foo()]
            pub struct B<|>ar
            ",
            &["pub struct Bar\n```\n___\n\n[Foo](https://docs.rs/test/*/test/struct.Foo.html)"],
        );
    }

    #[test]
    fn test_hover_intra_link_shortlink_namspaced_code() {
        check_hover_result(
            r"
            //- /lib.rs
            pub struct Foo;
            /// [`struct Foo`]
            pub struct B<|>ar
            ",
            &["pub struct Bar\n```\n___\n\n[`Foo`](https://docs.rs/test/*/test/struct.Foo.html)"],
        );
    }

    #[test]
    fn test_hover_intra_link_shortlink_namspaced_code_with_at() {
        check_hover_result(
            r"
            //- /lib.rs
            pub struct Foo;
            /// [`struct@Foo`]
            pub struct B<|>ar
            ",
            &["pub struct Bar\n```\n___\n\n[`Foo`](https://docs.rs/test/*/test/struct.Foo.html)"],
        );
    }

    #[test]
    fn test_hover_intra_link_reference() {
        check_hover_result(
            r"
            //- /lib.rs
            pub struct Foo;
            /// [my Foo][foo]
            ///
            /// [foo]: Foo
            pub struct B<|>ar
            ",
            &["pub struct Bar\n```\n___\n\n[my Foo](https://docs.rs/test/*/test/struct.Foo.html)"],
        );
    }

    #[test]
    fn test_hover_external_url() {
        check_hover_result(
            r"
            //- /lib.rs
            pub struct Foo;
            /// [external](https://www.google.com)
            pub struct B<|>ar
            ",
            &["pub struct Bar\n```\n___\n\n[external](https://www.google.com)"],
        );
    }

    // Check that we don't rewrite links which we can't identify
    #[test]
    fn test_hover_unknown_target() {
        check_hover_result(
            r"
            //- /lib.rs
            pub struct Foo;
            /// [baz](Baz)
            pub struct B<|>ar
            ",
            &["pub struct Bar\n```\n___\n\n[baz](Baz)"],
        );
    }

    #[test]
    fn test_hover_macro_generated_struct_fn_doc_comment() {
        mark::check!(hover_macro_generated_struct_fn_doc_comment);

        check_hover_result(
            r#"
            //- /lib.rs
            macro_rules! bar {
                () => {
                    struct Bar;
                    impl Bar {
                        /// Do the foo
                        fn foo(&self) {}
                    }
                }
            }

            bar!();

            fn foo() {
                let bar = Bar;
                bar.fo<|>o();
            }
            "#,
            &["test::Bar\n```\n\n```rust\nfn foo(&self)\n```\n___\n\nDo the foo"],
        );
    }

    #[test]
    fn test_hover_macro_generated_struct_fn_doc_attr() {
        mark::check!(hover_macro_generated_struct_fn_doc_attr);

        check_hover_result(
            r#"
            //- /lib.rs
            macro_rules! bar {
                () => {
                    struct Bar;
                    impl Bar {
                        #[doc = "Do the foo"]
                        fn foo(&self) {}
                    }
                }
            }

            bar!();

            fn foo() {
                let bar = Bar;
                bar.fo<|>o();
            }
            "#,
            &["test::Bar\n```\n\n```rust\nfn foo(&self)\n```\n___\n\nDo the foo"],
        );
    }

    #[test]
    fn test_hover_trait_has_impl_action() {
        let (_, actions) = check_hover_result(
            "
            //- /lib.rs
            trait foo<|>() {}
            ",
            &["trait foo"],
        );
        assert_impl_action(&actions[0], 6);
    }

    #[test]
    fn test_hover_struct_has_impl_action() {
        let (_, actions) = check_hover_result(
            "
            //- /lib.rs
            struct foo<|>() {}
            ",
            &["struct foo"],
        );
        assert_impl_action(&actions[0], 7);
    }

    #[test]
    fn test_hover_union_has_impl_action() {
        let (_, actions) = check_hover_result(
            "
            //- /lib.rs
            union foo<|>() {}
            ",
            &["union foo"],
        );
        assert_impl_action(&actions[0], 6);
    }

    #[test]
    fn test_hover_enum_has_impl_action() {
        let (_, actions) = check_hover_result(
            "
            //- /lib.rs
            enum foo<|>() {
                A,
                B
            }
            ",
            &["enum foo"],
        );
        assert_impl_action(&actions[0], 5);
    }

    #[test]
    fn test_hover_test_has_action() {
        let (_, actions) = check_hover_result(
            "
            //- /lib.rs
            #[test]
            fn foo_<|>test() {}
            ",
            &["fn foo_test()"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                Runnable(
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 0..24,
                            name: "foo_test",
                            kind: FN_DEF,
                            focus_range: Some(
                                11..19,
                            ),
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: Test {
                            test_id: Path(
                                "foo_test",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg_exprs: [],
                    },
                ),
            ]
            "###);
    }

    #[test]
    fn test_hover_test_mod_has_action() {
        let (_, actions) = check_hover_result(
            "
            //- /lib.rs
            mod tests<|> {
                #[test]
                fn foo_test() {}
            }
            ",
            &["mod tests"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                Runnable(
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 0..46,
                            name: "tests",
                            kind: MODULE,
                            focus_range: Some(
                                4..9,
                            ),
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: TestMod {
                            path: "tests",
                        },
                        cfg_exprs: [],
                    },
                ),
            ]
            "###);
    }

    #[test]
    fn test_hover_struct_has_goto_type_action() {
        let (_, actions) = check_hover_result(
            "
            //- /main.rs
            struct S{ f1: u32 }

            fn main() {
                let s<|>t = S{ f1:0 };
            }
            ",
            &["S"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..19,
                                name: "S",
                                kind: STRUCT_DEF,
                                focus_range: Some(
                                    7..8,
                                ),
                                container_name: None,
                                description: Some(
                                    "struct S",
                                ),
                                docs: None,
                            },
                        },
                    ],
                ),
            ]
            "###);
    }

    #[test]
    fn test_hover_generic_struct_has_goto_type_actions() {
        let (_, actions) = check_hover_result(
            "
            //- /main.rs
            struct Arg(u32);
            struct S<T>{ f1: T }

            fn main() {
                let s<|>t = S{ f1:Arg(0) };
            }
            ",
            &["S<Arg>"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 17..37,
                                name: "S",
                                kind: STRUCT_DEF,
                                focus_range: Some(
                                    24..25,
                                ),
                                container_name: None,
                                description: Some(
                                    "struct S",
                                ),
                                docs: None,
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "Arg",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..16,
                                name: "Arg",
                                kind: STRUCT_DEF,
                                focus_range: Some(
                                    7..10,
                                ),
                                container_name: None,
                                description: Some(
                                    "struct Arg",
                                ),
                                docs: None,
                            },
                        },
                    ],
                ),
            ]
            "###);
    }

    #[test]
    fn test_hover_generic_struct_has_flattened_goto_type_actions() {
        let (_, actions) = check_hover_result(
            "
            //- /main.rs
            struct Arg(u32);
            struct S<T>{ f1: T }

            fn main() {
                let s<|>t = S{ f1: S{ f1: Arg(0) } };
            }
            ",
            &["S<S<Arg>>"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 17..37,
                                name: "S",
                                kind: STRUCT_DEF,
                                focus_range: Some(
                                    24..25,
                                ),
                                container_name: None,
                                description: Some(
                                    "struct S",
                                ),
                                docs: None,
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "Arg",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..16,
                                name: "Arg",
                                kind: STRUCT_DEF,
                                focus_range: Some(
                                    7..10,
                                ),
                                container_name: None,
                                description: Some(
                                    "struct Arg",
                                ),
                                docs: None,
                            },
                        },
                    ],
                ),
            ]
            "###);
    }

    #[test]
    fn test_hover_tuple_has_goto_type_actions() {
        let (_, actions) = check_hover_result(
            "
            //- /main.rs
            struct A(u32);
            struct B(u32);
            mod M {
                pub struct C(u32);
            }

            fn main() {
                let s<|>t = (A(1), B(2), M::C(3) );
            }
            ",
            &["(A, B, C)"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "A",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..14,
                                name: "A",
                                kind: STRUCT_DEF,
                                focus_range: Some(
                                    7..8,
                                ),
                                container_name: None,
                                description: Some(
                                    "struct A",
                                ),
                                docs: None,
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "B",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 15..29,
                                name: "B",
                                kind: STRUCT_DEF,
                                focus_range: Some(
                                    22..23,
                                ),
                                container_name: None,
                                description: Some(
                                    "struct B",
                                ),
                                docs: None,
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "M::C",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 42..60,
                                name: "C",
                                kind: STRUCT_DEF,
                                focus_range: Some(
                                    53..54,
                                ),
                                container_name: None,
                                description: Some(
                                    "pub struct C",
                                ),
                                docs: None,
                            },
                        },
                    ],
                ),
            ]
               "###);
    }

    #[test]
    fn test_hover_return_impl_trait_has_goto_type_action() {
        let (_, actions) = check_hover_result(
            "
            //- /main.rs
            trait Foo {}

            fn foo() -> impl Foo {}

            fn main() {
                let s<|>t = foo();
            }
            ",
            &["impl Foo"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..12,
                                name: "Foo",
                                kind: TRAIT_DEF,
                                focus_range: Some(
                                    6..9,
                                ),
                                container_name: None,
                                description: Some(
                                    "trait Foo",
                                ),
                                docs: None,
                            },
                        },
                    ],
                ),
            ]
            "###);
    }

    #[test]
    fn test_hover_generic_return_impl_trait_has_goto_type_action() {
        let (_, actions) = check_hover_result(
            "
            //- /main.rs
            trait Foo<T> {}
            struct S;

            fn foo() -> impl Foo<S> {}

            fn main() {
                let s<|>t = foo();
            }
            ",
            &["impl Foo<S>"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..15,
                                name: "Foo",
                                kind: TRAIT_DEF,
                                focus_range: Some(
                                    6..9,
                                ),
                                container_name: None,
                                description: Some(
                                    "trait Foo",
                                ),
                                docs: None,
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 16..25,
                                name: "S",
                                kind: STRUCT_DEF,
                                focus_range: Some(
                                    23..24,
                                ),
                                container_name: None,
                                description: Some(
                                    "struct S",
                                ),
                                docs: None,
                            },
                        },
                    ],
                ),
            ]
            "###);
    }

    #[test]
    fn test_hover_return_impl_traits_has_goto_type_action() {
        let (_, actions) = check_hover_result(
            "
            //- /main.rs
            trait Foo {}
            trait Bar {}

            fn foo() -> impl Foo + Bar {}

            fn main() {
                let s<|>t = foo();
            }
            ",
            &["impl Foo + Bar"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..12,
                                name: "Foo",
                                kind: TRAIT_DEF,
                                focus_range: Some(
                                    6..9,
                                ),
                                container_name: None,
                                description: Some(
                                    "trait Foo",
                                ),
                                docs: None,
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "Bar",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 13..25,
                                name: "Bar",
                                kind: TRAIT_DEF,
                                focus_range: Some(
                                    19..22,
                                ),
                                container_name: None,
                                description: Some(
                                    "trait Bar",
                                ),
                                docs: None,
                            },
                        },
                    ],
                ),
            ]
            "###);
    }

    #[test]
    fn test_hover_generic_return_impl_traits_has_goto_type_action() {
        let (_, actions) = check_hover_result(
            "
            //- /main.rs
            trait Foo<T> {}
            trait Bar<T> {}
            struct S1 {}
            struct S2 {}

            fn foo() -> impl Foo<S1> + Bar<S2> {}

            fn main() {
                let s<|>t = foo();
            }
            ",
            &["impl Foo<S1> + Bar<S2>"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..15,
                                name: "Foo",
                                kind: TRAIT_DEF,
                                focus_range: Some(
                                    6..9,
                                ),
                                container_name: None,
                                description: Some(
                                    "trait Foo",
                                ),
                                docs: None,
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "Bar",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 16..31,
                                name: "Bar",
                                kind: TRAIT_DEF,
                                focus_range: Some(
                                    22..25,
                                ),
                                container_name: None,
                                description: Some(
                                    "trait Bar",
                                ),
                                docs: None,
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "S1",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 32..44,
                                name: "S1",
                                kind: STRUCT_DEF,
                                focus_range: Some(
                                    39..41,
                                ),
                                container_name: None,
                                description: Some(
                                    "struct S1",
                                ),
                                docs: None,
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "S2",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 45..57,
                                name: "S2",
                                kind: STRUCT_DEF,
                                focus_range: Some(
                                    52..54,
                                ),
                                container_name: None,
                                description: Some(
                                    "struct S2",
                                ),
                                docs: None,
                            },
                        },
                    ],
                ),
            ]
               "###);
    }

    #[test]
    fn test_hover_arg_impl_trait_has_goto_type_action() {
        let (_, actions) = check_hover_result(
            "
            //- /lib.rs
            trait Foo {}
            fn foo(ar<|>g: &impl Foo) {}
            ",
            &["&impl Foo"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..12,
                                name: "Foo",
                                kind: TRAIT_DEF,
                                focus_range: Some(
                                    6..9,
                                ),
                                container_name: None,
                                description: Some(
                                    "trait Foo",
                                ),
                                docs: None,
                            },
                        },
                    ],
                ),
            ]
            "###);
    }

    #[test]
    fn test_hover_arg_impl_traits_has_goto_type_action() {
        let (_, actions) = check_hover_result(
            "
            //- /lib.rs
            trait Foo {}
            trait Bar<T> {}
            struct S{}

            fn foo(ar<|>g: &impl Foo + Bar<S>) {}
            ",
            &["&impl Foo + Bar<S>"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..12,
                                name: "Foo",
                                kind: TRAIT_DEF,
                                focus_range: Some(
                                    6..9,
                                ),
                                container_name: None,
                                description: Some(
                                    "trait Foo",
                                ),
                                docs: None,
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "Bar",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 13..28,
                                name: "Bar",
                                kind: TRAIT_DEF,
                                focus_range: Some(
                                    19..22,
                                ),
                                container_name: None,
                                description: Some(
                                    "trait Bar",
                                ),
                                docs: None,
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 29..39,
                                name: "S",
                                kind: STRUCT_DEF,
                                focus_range: Some(
                                    36..37,
                                ),
                                container_name: None,
                                description: Some(
                                    "struct S",
                                ),
                                docs: None,
                            },
                        },
                    ],
                ),
            ]
            "###);
    }

    #[test]
    fn test_hover_arg_generic_impl_trait_has_goto_type_action() {
        let (_, actions) = check_hover_result(
            "
            //- /lib.rs
            trait Foo<T> {}
            struct S {}
            fn foo(ar<|>g: &impl Foo<S>) {}
            ",
            &["&impl Foo<S>"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..15,
                                name: "Foo",
                                kind: TRAIT_DEF,
                                focus_range: Some(
                                    6..9,
                                ),
                                container_name: None,
                                description: Some(
                                    "trait Foo",
                                ),
                                docs: None,
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 16..27,
                                name: "S",
                                kind: STRUCT_DEF,
                                focus_range: Some(
                                    23..24,
                                ),
                                container_name: None,
                                description: Some(
                                    "struct S",
                                ),
                                docs: None,
                            },
                        },
                    ],
                ),
            ]
            "###);
    }

    #[test]
    fn test_hover_dyn_return_has_goto_type_action() {
        let (_, actions) = check_hover_result(
            "
            //- /main.rs
            trait Foo {}
            struct S;
            impl Foo for S {}

            struct B<T>{}

            fn foo() -> B<dyn Foo> {}

            fn main() {
                let s<|>t = foo();
            }
            ",
            &["B<dyn Foo>"],
        );
        assert_debug_snapshot!(actions,
            @r###"
        [
            GoToType(
                [
                    HoverGotoTypeData {
                        mod_path: "B",
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 42..55,
                            name: "B",
                            kind: STRUCT_DEF,
                            focus_range: Some(
                                49..50,
                            ),
                            container_name: None,
                            description: Some(
                                "struct B",
                            ),
                            docs: None,
                        },
                    },
                    HoverGotoTypeData {
                        mod_path: "Foo",
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 0..12,
                            name: "Foo",
                            kind: TRAIT_DEF,
                            focus_range: Some(
                                6..9,
                            ),
                            container_name: None,
                            description: Some(
                                "trait Foo",
                            ),
                            docs: None,
                        },
                    },
                ],
            ),
        ]
        "###);
    }

    #[test]
    fn test_hover_dyn_arg_has_goto_type_action() {
        let (_, actions) = check_hover_result(
            "
            //- /lib.rs
            trait Foo {}
            fn foo(ar<|>g: &dyn Foo) {}
            ",
            &["&dyn Foo"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..12,
                                name: "Foo",
                                kind: TRAIT_DEF,
                                focus_range: Some(
                                    6..9,
                                ),
                                container_name: None,
                                description: Some(
                                    "trait Foo",
                                ),
                                docs: None,
                            },
                        },
                    ],
                ),
            ]
            "###);
    }

    #[test]
    fn test_hover_generic_dyn_arg_has_goto_type_action() {
        let (_, actions) = check_hover_result(
            "
            //- /lib.rs
            trait Foo<T> {}
            struct S {}
            fn foo(ar<|>g: &dyn Foo<S>) {}
            ",
            &["&dyn Foo<S>"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..15,
                                name: "Foo",
                                kind: TRAIT_DEF,
                                focus_range: Some(
                                    6..9,
                                ),
                                container_name: None,
                                description: Some(
                                    "trait Foo",
                                ),
                                docs: None,
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 16..27,
                                name: "S",
                                kind: STRUCT_DEF,
                                focus_range: Some(
                                    23..24,
                                ),
                                container_name: None,
                                description: Some(
                                    "struct S",
                                ),
                                docs: None,
                            },
                        },
                    ],
                ),
            ]
            "###);
    }

    #[test]
    fn test_hover_goto_type_action_links_order() {
        let (_, actions) = check_hover_result(
            "
            //- /lib.rs
            trait ImplTrait<T> {}
            trait DynTrait<T> {}
            struct B<T> {}
            struct S {}

            fn foo(a<|>rg: &impl ImplTrait<B<dyn DynTrait<B<S>>>>) {}
            ",
            &["&impl ImplTrait<B<dyn DynTrait<B<S>>>>"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ImplTrait",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..21,
                                name: "ImplTrait",
                                kind: TRAIT_DEF,
                                focus_range: Some(
                                    6..15,
                                ),
                                container_name: None,
                                description: Some(
                                    "trait ImplTrait",
                                ),
                                docs: None,
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "B",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 43..57,
                                name: "B",
                                kind: STRUCT_DEF,
                                focus_range: Some(
                                    50..51,
                                ),
                                container_name: None,
                                description: Some(
                                    "struct B",
                                ),
                                docs: None,
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "DynTrait",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 22..42,
                                name: "DynTrait",
                                kind: TRAIT_DEF,
                                focus_range: Some(
                                    28..36,
                                ),
                                container_name: None,
                                description: Some(
                                    "trait DynTrait",
                                ),
                                docs: None,
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 58..69,
                                name: "S",
                                kind: STRUCT_DEF,
                                focus_range: Some(
                                    65..66,
                                ),
                                container_name: None,
                                description: Some(
                                    "struct S",
                                ),
                                docs: None,
                            },
                        },
                    ],
                ),
            ]
            "###);
    }

    #[test]
    fn test_hover_associated_type_has_goto_type_action() {
        let (_, actions) = check_hover_result(
            "
            //- /main.rs
            trait Foo {
                type Item;
                fn get(self) -> Self::Item {}
            }

            struct Bar{}
            struct S{}

            impl Foo for S{
                type Item = Bar;
            }

            fn test() -> impl Foo {
                S{}
            }

            fn main() {
                let s<|>t = test().get();
            }
            ",
            &["Foo::Item<impl Foo>"],
        );
        assert_debug_snapshot!(actions,
            @r###"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..62,
                                name: "Foo",
                                kind: TRAIT_DEF,
                                focus_range: Some(
                                    6..9,
                                ),
                                container_name: None,
                                description: Some(
                                    "trait Foo",
                                ),
                                docs: None,
                            },
                        },
                    ],
                ),
            ]
            "###);
    }
}
