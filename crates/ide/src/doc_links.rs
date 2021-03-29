//! Extracts, resolves and rewrites links and intra-doc links in markdown documentation.

use std::{convert::TryFrom, iter::once, ops::Range};

use itertools::Itertools;
use pulldown_cmark::{BrokenLink, CowStr, Event, InlineStr, LinkType, Options, Parser, Tag};
use pulldown_cmark_to_cmark::{cmark_with_options, Options as CmarkOptions};
use url::Url;

use hir::{
    db::{DefDatabase, HirDatabase},
    Adt, AsAssocItem, AssocItem, AssocItemContainer, Crate, Field, HasAttrs, ItemInNs, ModuleDef,
};
use ide_db::{
    defs::{Definition, NameClass, NameRefClass},
    RootDatabase,
};
use syntax::{
    ast, match_ast, AstNode, AstToken, SyntaxKind::*, SyntaxNode, SyntaxToken, TextRange, TextSize,
    TokenAtOffset, T,
};

use crate::{FilePosition, Semantics};

pub(crate) type DocumentationLink = String;

/// Rewrite documentation links in markdown to point to an online host (e.g. docs.rs)
pub(crate) fn rewrite_links(db: &RootDatabase, markdown: &str, definition: &Definition) -> String {
    let mut cb = |link: BrokenLink| {
        Some((
            /*url*/ link.reference.to_owned().into(),
            /*title*/ link.reference.to_owned().into(),
        ))
    };
    let doc = Parser::new_with_broken_link_callback(markdown, Options::empty(), Some(&mut cb));

    let doc = map_links(doc, |target, title: &str| {
        // This check is imperfect, there's some overlap between valid intra-doc links
        // and valid URLs so we choose to be too eager to try to resolve what might be
        // a URL.
        if target.contains("://") {
            (target.to_string(), title.to_string())
        } else {
            // Two possibilities:
            // * path-based links: `../../module/struct.MyStruct.html`
            // * module-based links (AKA intra-doc links): `super::super::module::MyStruct`
            if let Some(rewritten) = rewrite_intra_doc_link(db, *definition, target, title) {
                return rewritten;
            }
            if let Definition::ModuleDef(def) = *definition {
                if let Some(target) = rewrite_url_link(db, def, target) {
                    return (target, title.to_string());
                }
            }

            (target.to_string(), title.to_string())
        }
    });
    let mut out = String::new();
    let mut options = CmarkOptions::default();
    options.code_block_backticks = 3;
    cmark_with_options(doc, &mut out, None, options).ok();
    out
}

/// Remove all links in markdown documentation.
pub(crate) fn remove_links(markdown: &str) -> String {
    let mut drop_link = false;

    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_FOOTNOTES);

    let mut cb = |_: BrokenLink| {
        let empty = InlineStr::try_from("").unwrap();
        Some((CowStr::Inlined(empty), CowStr::Inlined(empty)))
    };
    let doc = Parser::new_with_broken_link_callback(markdown, opts, Some(&mut cb));
    let doc = doc.filter_map(move |evt| match evt {
        Event::Start(Tag::Link(link_type, ref target, ref title)) => {
            if link_type == LinkType::Inline && target.contains("://") {
                Some(Event::Start(Tag::Link(link_type, target.clone(), title.clone())))
            } else {
                drop_link = true;
                None
            }
        }
        Event::End(_) if drop_link => {
            drop_link = false;
            None
        }
        _ => Some(evt),
    });

    let mut out = String::new();
    let mut options = CmarkOptions::default();
    options.code_block_backticks = 3;
    cmark_with_options(doc, &mut out, None, options).ok();
    out
}

pub(crate) fn extract_definitions_from_markdown(
    markdown: &str,
) -> Vec<(Range<usize>, String, Option<hir::Namespace>)> {
    let mut res = vec![];
    let mut cb = |link: BrokenLink| {
        // These allocations are actually unnecessary but the lifetimes on BrokenLinkCallback are wrong
        // this is fixed in the repo but not on the crates.io release yet
        Some((
            /*url*/ link.reference.to_owned().into(),
            /*title*/ link.reference.to_owned().into(),
        ))
    };
    let doc = Parser::new_with_broken_link_callback(markdown, Options::empty(), Some(&mut cb));
    for (event, range) in doc.into_offset_iter() {
        if let Event::Start(Tag::Link(_, target, title)) = event {
            let link = if target.is_empty() { title } else { target };
            let (link, ns) = parse_intra_doc_link(&link);
            res.push((range, link.to_string(), ns));
        }
    }
    res
}

/// Extracts a link from a comment at the given position returning the spanning range, link and
/// optionally it's namespace.
pub(crate) fn extract_positioned_link_from_comment(
    position: TextSize,
    comment: &ast::Comment,
) -> Option<(TextRange, String, Option<hir::Namespace>)> {
    let doc_comment = comment.doc_comment()?;
    let comment_start =
        comment.syntax().text_range().start() + TextSize::from(comment.prefix().len() as u32);
    let def_links = extract_definitions_from_markdown(doc_comment);
    let (range, def_link, ns) =
        def_links.into_iter().find_map(|(Range { start, end }, def_link, ns)| {
            let range = TextRange::at(
                comment_start + TextSize::from(start as u32),
                TextSize::from((end - start) as u32),
            );
            range.contains(position).then(|| (range, def_link, ns))
        })?;
    Some((range, def_link, ns))
}

/// Turns a syntax node into it's [`Definition`] if it can hold docs.
pub(crate) fn doc_owner_to_def(
    sema: &Semantics<RootDatabase>,
    item: &SyntaxNode,
) -> Option<Definition> {
    let res: hir::ModuleDef = match_ast! {
        match item {
            ast::SourceFile(_it) => sema.scope(item).module()?.into(),
            ast::Fn(it) => sema.to_def(&it)?.into(),
            ast::Struct(it) => sema.to_def(&it)?.into(),
            ast::Enum(it) => sema.to_def(&it)?.into(),
            ast::Union(it) => sema.to_def(&it)?.into(),
            ast::Trait(it) => sema.to_def(&it)?.into(),
            ast::Const(it) => sema.to_def(&it)?.into(),
            ast::Static(it) => sema.to_def(&it)?.into(),
            ast::TypeAlias(it) => sema.to_def(&it)?.into(),
            ast::Variant(it) => sema.to_def(&it)?.into(),
            ast::Trait(it) => sema.to_def(&it)?.into(),
            ast::Impl(it) => return sema.to_def(&it).map(Definition::SelfType),
            ast::Macro(it) => return sema.to_def(&it).map(Definition::Macro),
            ast::TupleField(it) => return sema.to_def(&it).map(Definition::Field),
            ast::RecordField(it) => return sema.to_def(&it).map(Definition::Field),
            _ => return None,
        }
    };
    Some(Definition::ModuleDef(res))
}

pub(crate) fn resolve_doc_path_for_def(
    db: &dyn HirDatabase,
    def: Definition,
    link: &str,
    ns: Option<hir::Namespace>,
) -> Option<hir::ModuleDef> {
    match def {
        Definition::ModuleDef(def) => match def {
            ModuleDef::Module(it) => it.resolve_doc_path(db, &link, ns),
            ModuleDef::Function(it) => it.resolve_doc_path(db, &link, ns),
            ModuleDef::Adt(it) => it.resolve_doc_path(db, &link, ns),
            ModuleDef::Variant(it) => it.resolve_doc_path(db, &link, ns),
            ModuleDef::Const(it) => it.resolve_doc_path(db, &link, ns),
            ModuleDef::Static(it) => it.resolve_doc_path(db, &link, ns),
            ModuleDef::Trait(it) => it.resolve_doc_path(db, &link, ns),
            ModuleDef::TypeAlias(it) => it.resolve_doc_path(db, &link, ns),
            ModuleDef::BuiltinType(_) => None,
        },
        Definition::Macro(it) => it.resolve_doc_path(db, &link, ns),
        Definition::Field(it) => it.resolve_doc_path(db, &link, ns),
        Definition::SelfType(_)
        | Definition::Local(_)
        | Definition::GenericParam(_)
        | Definition::Label(_) => None,
    }
}

// FIXME:
// BUG: For Option::Some
// Returns https://doc.rust-lang.org/nightly/core/prelude/v1/enum.Option.html#variant.Some
// Instead of https://doc.rust-lang.org/nightly/core/option/enum.Option.html
//
// This should cease to be a problem if RFC2988 (Stable Rustdoc URLs) is implemented
// https://github.com/rust-lang/rfcs/pull/2988
fn get_doc_link(db: &RootDatabase, definition: Definition) -> Option<String> {
    // Get the outermost definition for the moduledef. This is used to resolve the public path to the type,
    // then we can join the method, field, etc onto it if required.
    let target_def: ModuleDef = match definition {
        Definition::ModuleDef(moddef) => match moddef {
            ModuleDef::Function(f) => f
                .as_assoc_item(db)
                .and_then(|assoc| match assoc.container(db) {
                    AssocItemContainer::Trait(t) => Some(t.into()),
                    AssocItemContainer::Impl(impld) => {
                        impld.self_ty(db).as_adt().map(|adt| adt.into())
                    }
                })
                .unwrap_or_else(|| f.clone().into()),
            moddef => moddef,
        },
        Definition::Field(f) => f.parent_def(db).into(),
        // FIXME: Handle macros
        _ => return None,
    };

    let ns = ItemInNs::from(target_def);

    let module = definition.module(db)?;
    let krate = module.krate();
    let import_map = db.import_map(krate.into());
    let base = once(krate.display_name(db)?.to_string())
        .chain(import_map.path_of(ns)?.segments.iter().map(|name| name.to_string()))
        .join("/")
        + "/";

    let filename = get_symbol_filename(db, &target_def);
    let fragment = match definition {
        Definition::ModuleDef(moddef) => match moddef {
            ModuleDef::Function(f) => {
                get_symbol_fragment(db, &FieldOrAssocItem::AssocItem(AssocItem::Function(f)))
            }
            ModuleDef::Const(c) => {
                get_symbol_fragment(db, &FieldOrAssocItem::AssocItem(AssocItem::Const(c)))
            }
            ModuleDef::TypeAlias(ty) => {
                get_symbol_fragment(db, &FieldOrAssocItem::AssocItem(AssocItem::TypeAlias(ty)))
            }
            _ => None,
        },
        Definition::Field(field) => get_symbol_fragment(db, &FieldOrAssocItem::Field(field)),
        _ => None,
    };

    get_doc_url(db, &krate)?
        .join(&base)
        .ok()
        .and_then(|mut url| {
            if !matches!(definition, Definition::ModuleDef(ModuleDef::Module(..))) {
                url.path_segments_mut().ok()?.pop();
            };
            Some(url)
        })
        .and_then(|url| url.join(filename.as_deref()?).ok())
        .and_then(
            |url| if let Some(fragment) = fragment { url.join(&fragment).ok() } else { Some(url) },
        )
        .map(|url| url.into_string())
}

fn rewrite_intra_doc_link(
    db: &RootDatabase,
    def: Definition,
    target: &str,
    title: &str,
) -> Option<(String, String)> {
    let link = if target.is_empty() { title } else { target };
    let (link, ns) = parse_intra_doc_link(link);
    let resolved = resolve_doc_path_for_def(db, def, link, ns)?;
    let krate = resolved.module(db)?.krate();
    let canonical_path = resolved.canonical_path(db)?;
    let mut new_url = get_doc_url(db, &krate)?
        .join(&format!("{}/", krate.display_name(db)?))
        .ok()?
        .join(&canonical_path.replace("::", "/"))
        .ok()?
        .join(&get_symbol_filename(db, &resolved)?)
        .ok()?;

    if let ModuleDef::Trait(t) = resolved {
        if let Some(assoc_item) = t.items(db).into_iter().find_map(|assoc_item| {
            if let Some(name) = assoc_item.name(db) {
                if *link == format!("{}::{}", canonical_path, name) {
                    return Some(assoc_item);
                }
            }
            None
        }) {
            if let Some(fragment) =
                get_symbol_fragment(db, &FieldOrAssocItem::AssocItem(assoc_item))
            {
                new_url = new_url.join(&fragment).ok()?;
            }
        };
    }

    Some((new_url.into_string(), strip_prefixes_suffixes(title).to_string()))
}

/// Try to resolve path to local documentation via path-based links (i.e. `../gateway/struct.Shard.html`).
fn rewrite_url_link(db: &RootDatabase, def: ModuleDef, target: &str) -> Option<String> {
    if !(target.contains('#') || target.contains(".html")) {
        return None;
    }

    let module = def.module(db)?;
    let krate = module.krate();
    let canonical_path = def.canonical_path(db)?;
    let base = format!("{}/{}", krate.display_name(db)?, canonical_path.replace("::", "/"));

    get_doc_url(db, &krate)
        .and_then(|url| url.join(&base).ok())
        .and_then(|url| {
            get_symbol_filename(db, &def).as_deref().map(|f| url.join(f).ok()).flatten()
        })
        .and_then(|url| url.join(target).ok())
        .map(|url| url.into_string())
}

/// Retrieve a link to documentation for the given symbol.
pub(crate) fn external_docs(
    db: &RootDatabase,
    position: &FilePosition,
) -> Option<DocumentationLink> {
    let sema = Semantics::new(db);
    let file = sema.parse(position.file_id).syntax().clone();
    let token = pick_best(file.token_at_offset(position.offset))?;
    let token = sema.descend_into_macros(token);

    let node = token.parent()?;
    let definition = match_ast! {
        match node {
            ast::NameRef(name_ref) => NameRefClass::classify(&sema, &name_ref).map(|d| d.referenced(sema.db)),
            ast::Name(name) => NameClass::classify(&sema, &name).map(|d| d.referenced_or_defined(sema.db)),
            _ => None,
        }
    };

    get_doc_link(db, definition?)
}

/// Rewrites a markdown document, applying 'callback' to each link.
fn map_links<'e>(
    events: impl Iterator<Item = Event<'e>>,
    callback: impl Fn(&str, &str) -> (String, String),
) -> impl Iterator<Item = Event<'e>> {
    let mut in_link = false;
    let mut link_target: Option<CowStr> = None;

    events.map(move |evt| match evt {
        Event::Start(Tag::Link(_link_type, ref target, _)) => {
            in_link = true;
            link_target = Some(target.clone());
            evt
        }
        Event::End(Tag::Link(link_type, _target, _)) => {
            in_link = false;
            Event::End(Tag::Link(link_type, link_target.take().unwrap(), CowStr::Borrowed("")))
        }
        Event::Text(s) if in_link => {
            let (link_target_s, link_name) = callback(&link_target.take().unwrap(), &s);
            link_target = Some(CowStr::Boxed(link_target_s.into()));
            Event::Text(CowStr::Boxed(link_name.into()))
        }
        Event::Code(s) if in_link => {
            let (link_target_s, link_name) = callback(&link_target.take().unwrap(), &s);
            link_target = Some(CowStr::Boxed(link_target_s.into()));
            Event::Code(CowStr::Boxed(link_name.into()))
        }
        _ => evt,
    })
}

const TYPES: ([&str; 9], [&str; 0]) =
    (["type", "struct", "enum", "mod", "trait", "union", "module", "prim", "primitive"], []);
const VALUES: ([&str; 8], [&str; 1]) =
    (["value", "function", "fn", "method", "const", "static", "mod", "module"], ["()"]);
const MACROS: ([&str; 2], [&str; 1]) = (["macro", "derive"], ["!"]);

/// Extract the specified namespace from an intra-doc-link if one exists.
///
/// # Examples
///
/// * `struct MyStruct` -> ("MyStruct", `Namespace::Types`)
/// * `panic!` -> ("panic", `Namespace::Macros`)
/// * `fn@from_intra_spec` -> ("from_intra_spec", `Namespace::Values`)
fn parse_intra_doc_link(s: &str) -> (&str, Option<hir::Namespace>) {
    let s = s.trim_matches('`');

    [
        (hir::Namespace::Types, (TYPES.0.iter(), TYPES.1.iter())),
        (hir::Namespace::Values, (VALUES.0.iter(), VALUES.1.iter())),
        (hir::Namespace::Macros, (MACROS.0.iter(), MACROS.1.iter())),
    ]
    .iter()
    .cloned()
    .find_map(|(ns, (mut prefixes, mut suffixes))| {
        if let Some(prefix) = prefixes.find(|&&prefix| {
            s.starts_with(prefix)
                && s.chars().nth(prefix.len()).map_or(false, |c| c == '@' || c == ' ')
        }) {
            Some((&s[prefix.len() + 1..], ns))
        } else {
            suffixes.find_map(|&suffix| s.strip_suffix(suffix).zip(Some(ns)))
        }
    })
    .map_or((s, None), |(s, ns)| (s, Some(ns)))
}

fn strip_prefixes_suffixes(s: &str) -> &str {
    [
        (TYPES.0.iter(), TYPES.1.iter()),
        (VALUES.0.iter(), VALUES.1.iter()),
        (MACROS.0.iter(), MACROS.1.iter()),
    ]
    .iter()
    .cloned()
    .find_map(|(mut prefixes, mut suffixes)| {
        if let Some(prefix) = prefixes.find(|&&prefix| {
            s.starts_with(prefix)
                && s.chars().nth(prefix.len()).map_or(false, |c| c == '@' || c == ' ')
        }) {
            Some(&s[prefix.len() + 1..])
        } else {
            suffixes.find_map(|&suffix| s.strip_suffix(suffix))
        }
    })
    .unwrap_or(s)
}

/// Get the root URL for the documentation of a crate.
///
/// ```
/// https://doc.rust-lang.org/std/iter/trait.Iterator.html#tymethod.next
/// ^^^^^^^^^^^^^^^^^^^^^^^^^^
/// ```
fn get_doc_url(db: &RootDatabase, krate: &Crate) -> Option<Url> {
    krate
        .get_html_root_url(db)
        .or_else(|| {
            // Fallback to docs.rs. This uses `display_name` and can never be
            // correct, but that's what fallbacks are about.
            //
            // FIXME: clicking on the link should just open the file in the editor,
            // instead of falling back to external urls.
            Some(format!("https://docs.rs/{}/*/", krate.display_name(db)?))
        })
        .and_then(|s| Url::parse(&s).ok())
}

/// Get the filename and extension generated for a symbol by rustdoc.
///
/// ```
/// https://doc.rust-lang.org/std/iter/trait.Iterator.html#tymethod.next
///                                    ^^^^^^^^^^^^^^^^^^^
/// ```
fn get_symbol_filename(db: &dyn HirDatabase, definition: &ModuleDef) -> Option<String> {
    Some(match definition {
        ModuleDef::Adt(adt) => match adt {
            Adt::Struct(s) => format!("struct.{}.html", s.name(db)),
            Adt::Enum(e) => format!("enum.{}.html", e.name(db)),
            Adt::Union(u) => format!("union.{}.html", u.name(db)),
        },
        ModuleDef::Module(_) => "index.html".to_string(),
        ModuleDef::Trait(t) => format!("trait.{}.html", t.name(db)),
        ModuleDef::TypeAlias(t) => format!("type.{}.html", t.name(db)),
        ModuleDef::BuiltinType(t) => format!("primitive.{}.html", t.name()),
        ModuleDef::Function(f) => format!("fn.{}.html", f.name(db)),
        ModuleDef::Variant(ev) => {
            format!("enum.{}.html#variant.{}", ev.parent_enum(db).name(db), ev.name(db))
        }
        ModuleDef::Const(c) => format!("const.{}.html", c.name(db)?),
        ModuleDef::Static(s) => format!("static.{}.html", s.name(db)?),
    })
}

enum FieldOrAssocItem {
    Field(Field),
    AssocItem(AssocItem),
}

/// Get the fragment required to link to a specific field, method, associated type, or associated constant.
///
/// ```
/// https://doc.rust-lang.org/std/iter/trait.Iterator.html#tymethod.next
///                                                       ^^^^^^^^^^^^^^
/// ```
fn get_symbol_fragment(db: &dyn HirDatabase, field_or_assoc: &FieldOrAssocItem) -> Option<String> {
    Some(match field_or_assoc {
        FieldOrAssocItem::Field(field) => format!("#structfield.{}", field.name(db)),
        FieldOrAssocItem::AssocItem(assoc) => match assoc {
            AssocItem::Function(function) => {
                let is_trait_method = function
                    .as_assoc_item(db)
                    .and_then(|assoc| assoc.containing_trait(db))
                    .is_some();
                // This distinction may get more complicated when specialization is available.
                // Rustdoc makes this decision based on whether a method 'has defaultness'.
                // Currently this is only the case for provided trait methods.
                if is_trait_method && !function.has_body(db) {
                    format!("#tymethod.{}", function.name(db))
                } else {
                    format!("#method.{}", function.name(db))
                }
            }
            AssocItem::Const(constant) => format!("#associatedconstant.{}", constant.name(db)?),
            AssocItem::TypeAlias(ty) => format!("#associatedtype.{}", ty.name(db)),
        },
    })
}

fn pick_best(tokens: TokenAtOffset<SyntaxToken>) -> Option<SyntaxToken> {
    return tokens.max_by_key(priority);
    fn priority(n: &SyntaxToken) -> usize {
        match n.kind() {
            IDENT | INT_NUMBER => 3,
            T!['('] | T![')'] => 2,
            kind if kind.is_trivia() => 0,
            _ => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::fixture;

    fn check(ra_fixture: &str, expect: Expect) {
        let (analysis, position) = fixture::position(ra_fixture);
        let url = analysis.external_docs(position).unwrap().expect("could not find url for symbol");

        expect.assert_eq(&url)
    }

    #[test]
    fn test_doc_url_struct() {
        check(
            r#"
pub struct Fo$0o;
"#,
            expect![[r#"https://docs.rs/test/*/test/struct.Foo.html"#]],
        );
    }

    #[test]
    fn test_doc_url_fn() {
        check(
            r#"
pub fn fo$0o() {}
"#,
            expect![[r##"https://docs.rs/test/*/test/fn.foo.html#method.foo"##]],
        );
    }

    #[test]
    fn test_doc_url_inherent_method() {
        check(
            r#"
pub struct Foo;

impl Foo {
    pub fn met$0hod() {}
}

"#,
            expect![[r##"https://docs.rs/test/*/test/struct.Foo.html#method.method"##]],
        );
    }

    #[test]
    fn test_doc_url_trait_provided_method() {
        check(
            r#"
pub trait Bar {
    fn met$0hod() {}
}

"#,
            expect![[r##"https://docs.rs/test/*/test/trait.Bar.html#method.method"##]],
        );
    }

    #[test]
    fn test_doc_url_trait_required_method() {
        check(
            r#"
pub trait Foo {
    fn met$0hod();
}

"#,
            expect![[r##"https://docs.rs/test/*/test/trait.Foo.html#tymethod.method"##]],
        );
    }

    #[test]
    fn test_doc_url_field() {
        check(
            r#"
pub struct Foo {
    pub fie$0ld: ()
}

"#,
            expect![[r##"https://docs.rs/test/*/test/struct.Foo.html#structfield.field"##]],
        );
    }

    #[test]
    fn test_module() {
        check(
            r#"
pub mod foo {
    pub mod ba$0r {}
}
        "#,
            expect![[r#"https://docs.rs/test/*/test/foo/bar/index.html"#]],
        )
    }

    // FIXME: ImportMap will return re-export paths instead of public module
    // paths. The correct path to documentation will never be a re-export.
    // This problem stops us from resolving stdlib items included in the prelude
    // such as `Option::Some` correctly.
    #[ignore = "ImportMap may return re-exports"]
    #[test]
    fn test_reexport_order() {
        check(
            r#"
pub mod wrapper {
    pub use module::Item;

    pub mod module {
        pub struct Item;
    }
}

fn foo() {
    let bar: wrapper::It$0em;
}
        "#,
            expect![[r#"https://docs.rs/test/*/test/wrapper/module/struct.Item.html"#]],
        )
    }
}
