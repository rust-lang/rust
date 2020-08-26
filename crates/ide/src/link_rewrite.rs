//! Resolves and rewrites links in markdown documentation.
//!
//! Most of the implementation can be found in [`hir::doc_links`].

use hir::{Adt, Crate, HasAttrs, ModuleDef};
use ide_db::{defs::Definition, RootDatabase};
use pulldown_cmark::{CowStr, Event, Options, Parser, Tag};
use pulldown_cmark_to_cmark::{cmark_with_options, Options as CmarkOptions};
use url::Url;

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
        .join(&format!("{}/", krate.display_name(db)?))
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
    if !(target.contains("#") || target.contains(".html")) {
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

// Rewrites a markdown document, resolving links using `callback` and additionally striping prefixes/suffixes on link titles.
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
    s.trim_start_matches("@").trim()
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
/// Example: `struct.Shard.html`
fn get_symbol_filename(db: &RootDatabase, definition: &ModuleDef) -> Option<String> {
    Some(match definition {
        ModuleDef::Adt(adt) => match adt {
            Adt::Struct(s) => format!("struct.{}.html", s.name(db)),
            Adt::Enum(e) => format!("enum.{}.html", e.name(db)),
            Adt::Union(u) => format!("union.{}.html", u.name(db)),
        },
        ModuleDef::Module(_) => "index.html".to_string(),
        ModuleDef::Trait(t) => format!("trait.{}.html", t.name(db)),
        ModuleDef::TypeAlias(t) => format!("type.{}.html", t.name(db)),
        ModuleDef::BuiltinType(t) => format!("primitive.{}.html", t),
        ModuleDef::Function(f) => format!("fn.{}.html", f.name(db)),
        ModuleDef::EnumVariant(ev) => {
            format!("enum.{}.html#variant.{}", ev.parent_enum(db).name(db), ev.name(db))
        }
        ModuleDef::Const(c) => format!("const.{}.html", c.name(db)?),
        ModuleDef::Static(s) => format!("static.{}.html", s.name(db)?),
    })
}
