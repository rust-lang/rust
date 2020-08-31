//! Resolves and rewrites links in markdown documentation.
//!
//! Most of the implementation can be found in [`hir::doc_links`].

use hir::{Adt, Crate, HasAttrs, ModuleDef};
use ide_db::{defs::Definition, RootDatabase};
use pulldown_cmark::{CowStr, Event, LinkType, Options, Parser, Tag};
use pulldown_cmark_to_cmark::{cmark_with_options, Options as CmarkOptions};
use url::Url;

use crate::{FilePosition, Semantics};
use hir::{get_doc_link, resolve_doc_link};
use ide_db::{
    defs::{classify_name, classify_name_ref, Definition},
    RootDatabase,
};
use syntax::{ast, match_ast, AstNode, SyntaxKind::*, SyntaxToken, TokenAtOffset, T};

pub type DocumentationLink = String;

/// Rewrite documentation links in markdown to point to an online host (e.g. docs.rs)
pub fn rewrite_links(db: &RootDatabase, markdown: &str, definition: &Definition) -> String {
    let doc = Parser::new_with_broken_link_callback(
        markdown,
        Options::empty(),
        Some(&|label, _| Some((/*url*/ label.to_string(), /*title*/ label.to_string()))),
    );

    let doc = map_links(doc, |target, title: &str| {
        // This check is imperfect, there's some overlap between valid intra-doc links
        // and valid URLs so we choose to be too eager to try to resolve what might be
        // a URL.
        if target.contains("://") {
            (target.to_string(), title.to_string())
        } else {
            // Two posibilities:
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
pub fn remove_links(markdown: &str) -> String {
    let mut drop_link = false;

    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_FOOTNOTES);

    let doc = Parser::new_with_broken_link_callback(
        markdown,
        opts,
        Some(&|_, _| Some((String::new(), String::new()))),
    );
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

pub fn get_doc_link<T: Resolvable + Clone>(db: &dyn HirDatabase, definition: &T) -> Option<String> {
    let module_def = definition.clone().try_into_module_def()?;

    get_doc_link_impl(db, &module_def)
}

// TODO:
// BUG: For Option
// Returns https://doc.rust-lang.org/nightly/core/prelude/v1/enum.Option.html#variant.Some
// Instead of https://doc.rust-lang.org/nightly/core/option/enum.Option.html
fn get_doc_link_impl(db: &dyn HirDatabase, moddef: &ModuleDef) -> Option<String> {
    // Get the outermost definition for the moduledef. This is used to resolve the public path to the type,
    // then we can join the method, field, etc onto it if required.
    let target_def: ModuleDef = match moddef {
        ModuleDef::Function(f) => match f.as_assoc_item(db).map(|assoc| assoc.container(db)) {
            Some(AssocItemContainer::Trait(t)) => t.into(),
            Some(AssocItemContainer::ImplDef(imp)) => {
                let resolver = ModuleId::from(imp.module(db)).resolver(db.upcast());
                let ctx = TyLoweringContext::new(db, &resolver);
                Adt::from(
                    Ty::from_hir(
                        &ctx,
                        &imp.target_trait(db).unwrap_or_else(|| imp.target_type(db)),
                    )
                    .as_adt()
                    .map(|t| t.0)
                    .unwrap(),
                )
                .into()
            }
            None => ModuleDef::Function(*f),
        },
        moddef => *moddef,
    };

    let ns = ItemInNs::Types(target_def.clone().into());

    let module = moddef.module(db)?;
    let krate = module.krate();
    let import_map = db.import_map(krate.into());
    let base = once(krate.display_name(db).unwrap())
        .chain(import_map.path_of(ns).unwrap().segments.iter().map(|name| format!("{}", name)))
        .join("/");

    get_doc_url(db, &krate)
        .and_then(|url| url.join(&base).ok())
        .and_then(|url| {
            get_symbol_filename(db, &target_def).as_deref().and_then(|f| url.join(f).ok())
        })
        .and_then(|url| match moddef {
            ModuleDef::Function(f) => {
                get_symbol_fragment(db, &FieldOrAssocItem::AssocItem(AssocItem::Function(*f)))
                    .as_deref()
                    .and_then(|f| url.join(f).ok())
            }
            ModuleDef::Const(c) => {
                get_symbol_fragment(db, &FieldOrAssocItem::AssocItem(AssocItem::Const(*c)))
                    .as_deref()
                    .and_then(|f| url.join(f).ok())
            }
            ModuleDef::TypeAlias(ty) => {
                get_symbol_fragment(db, &FieldOrAssocItem::AssocItem(AssocItem::TypeAlias(*ty)))
                    .as_deref()
                    .and_then(|f| url.join(f).ok())
            }
            // TODO:  Field <- this requires passing in a definition or something
            _ => Some(url),
        })
        .map(|url| url.into_string())
}

fn rewrite_intra_doc_link(
    db: &RootDatabase,
    def: Definition,
    target: &str,
    title: &str,
) -> Option<(String, String)> {
    let link = if target.is_empty() { title } else { target };
    let (link, ns) = parse_link(link);
    let resolved = match def {
        Definition::ModuleDef(def) => match def {
            ModuleDef::Module(it) => it.resolve_doc_path(db, link, ns),
            ModuleDef::Function(it) => it.resolve_doc_path(db, link, ns),
            ModuleDef::Adt(it) => it.resolve_doc_path(db, link, ns),
            ModuleDef::EnumVariant(it) => it.resolve_doc_path(db, link, ns),
            ModuleDef::Const(it) => it.resolve_doc_path(db, link, ns),
            ModuleDef::Static(it) => it.resolve_doc_path(db, link, ns),
            ModuleDef::Trait(it) => it.resolve_doc_path(db, link, ns),
            ModuleDef::TypeAlias(it) => it.resolve_doc_path(db, link, ns),
            ModuleDef::BuiltinType(_) => return None,
        },
        Definition::Macro(it) => it.resolve_doc_path(db, link, ns),
        Definition::Field(it) => it.resolve_doc_path(db, link, ns),
        Definition::SelfType(_) | Definition::Local(_) | Definition::TypeParam(_) => return None,
    }?;
    let krate = resolved.module(db)?.krate();
    let canonical_path = resolved.canonical_path(db)?;
    let new_target = get_doc_url(db, &krate)?
        .join(&format!("{}/", krate.declaration_name(db)?))
        .ok()?
        .join(&canonical_path.replace("::", "/"))
        .ok()?
        .join(&get_symbol_filename(db, &resolved)?)
        .ok()?
        .into_string();
    let new_title = strip_prefixes_suffixes(title);
    Some((new_target, new_title.to_string()))
}

/// Try to resolve path to local documentation via path-based links (i.e. `../gateway/struct.Shard.html`).
fn rewrite_url_link(db: &RootDatabase, def: ModuleDef, target: &str) -> Option<String> {
    if !(target.contains('#') || target.contains(".html")) {
        return None;
    }

    let module = def.module(db)?;
    let krate = module.krate();
    let canonical_path = def.canonical_path(db)?;
    let base = format!("{}/{}", krate.declaration_name(db)?, canonical_path.replace("::", "/"));

    get_doc_url(db, &krate)
        .and_then(|url| url.join(&base).ok())
        .and_then(|url| {
            get_symbol_filename(db, &def).as_deref().map(|f| url.join(f).ok()).flatten()
        })
        .and_then(|url| url.join(target).ok())
        .map(|url| url.into_string())
}

// FIXME: This should either be moved, or the module should be renamed.
/// Retrieve a link to documentation for the given symbol.
pub fn get_doc_url(db: &RootDatabase, position: &FilePosition) -> Option<DocumentationLink> {
    let sema = Semantics::new(db);
    let file = sema.parse(position.file_id).syntax().clone();
    let token = pick_best(file.token_at_offset(position.offset))?;
    let token = sema.descend_into_macros(token);

    let node = token.parent();
    let definition = match_ast! {
        match node {
            ast::NameRef(name_ref) => classify_name_ref(&sema, &name_ref).map(|d| d.definition(sema.db)),
            ast::Name(name) => classify_name(&sema, &name).map(|d| d.definition(sema.db)),
            _ => None,
        }
    };

    match definition? {
        Definition::Macro(t) => get_doc_link(db, &t),
        Definition::Field(t) => get_doc_link(db, &t),
        Definition::ModuleDef(t) => get_doc_link(db, &t),
        Definition::SelfType(t) => get_doc_link(db, &t),
        Definition::Local(t) => get_doc_link(db, &t),
        Definition::TypeParam(t) => get_doc_link(db, &t),
    }
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

fn parse_link(s: &str) -> (&str, Option<hir::Namespace>) {
    let path = strip_prefixes_suffixes(s);
    let ns = ns_from_intra_spec(s);
    (path, ns)
}

/// Strip prefixes, suffixes, and inline code marks from the given string.
fn strip_prefixes_suffixes(mut s: &str) -> &str {
    s = s.trim_matches('`');

    [
        (TYPES.0.iter(), TYPES.1.iter()),
        (VALUES.0.iter(), VALUES.1.iter()),
        (MACROS.0.iter(), MACROS.1.iter()),
    ]
    .iter()
    .for_each(|(prefixes, suffixes)| {
        prefixes.clone().for_each(|prefix| s = s.trim_start_matches(*prefix));
        suffixes.clone().for_each(|suffix| s = s.trim_end_matches(*suffix));
    });
    s.trim_start_matches('@').trim()
}

static TYPES: ([&str; 7], [&str; 0]) =
    (["type", "struct", "enum", "mod", "trait", "union", "module"], []);
static VALUES: ([&str; 8], [&str; 1]) =
    (["value", "function", "fn", "method", "const", "static", "mod", "module"], ["()"]);
static MACROS: ([&str; 1], [&str; 1]) = (["macro"], ["!"]);

/// Extract the specified namespace from an intra-doc-link if one exists.
///
/// # Examples
///
/// * `struct MyStruct` -> `Namespace::Types`
/// * `panic!` -> `Namespace::Macros`
/// * `fn@from_intra_spec` -> `Namespace::Values`
fn ns_from_intra_spec(s: &str) -> Option<hir::Namespace> {
    [
        (hir::Namespace::Types, (TYPES.0.iter(), TYPES.1.iter())),
        (hir::Namespace::Values, (VALUES.0.iter(), VALUES.1.iter())),
        (hir::Namespace::Macros, (MACROS.0.iter(), MACROS.1.iter())),
    ]
    .iter()
    .filter(|(_ns, (prefixes, suffixes))| {
        prefixes
            .clone()
            .map(|prefix| {
                s.starts_with(*prefix)
                    && s.chars()
                        .nth(prefix.len() + 1)
                        .map(|c| c == '@' || c == ' ')
                        .unwrap_or(false)
            })
            .any(|cond| cond)
            || suffixes
                .clone()
                .map(|suffix| {
                    s.starts_with(*suffix)
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
            Some(format!("https://docs.rs/{}/*/", krate.declaration_name(db)?))
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
        ModuleDef::BuiltinType(t) => format!("primitive.{}.html", t.as_name()),
        ModuleDef::Function(f) => format!("fn.{}.html", f.name(db)),
        ModuleDef::EnumVariant(ev) => {
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
            // TODO: Rustdoc sometimes uses tymethod instead of method. This case needs to be investigated.
            AssocItem::Function(function) => format!("#method.{}", function.name(db)),
            // TODO: This might be the old method for documenting associated constants, i32::MAX uses a separate page...
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
