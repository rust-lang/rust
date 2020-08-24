//! Resolves and rewrites links in markdown documentation for hovers/completion windows.

use std::iter::once;

use itertools::Itertools;
use url::Url;

use crate::{db::HirDatabase, Adt, AsName, Crate, Hygiene, ItemInNs, ModPath, ModuleDef};
use hir_def::{db::DefDatabase, resolver::Resolver};
use ra_syntax::ast::Path;

pub fn resolve_doc_link<T: Resolvable + Clone, D: DefDatabase + HirDatabase>(
    db: &D,
    definition: &T,
    link_text: &str,
    link_target: &str,
) -> Option<(String, String)> {
    try_resolve_intra(db, definition, link_text, &link_target).or_else(|| {
        if let Some(definition) = definition.clone().try_into_module_def() {
            try_resolve_path(db, &definition, &link_target)
                .map(|target| (target, link_text.to_string()))
        } else {
            None
        }
    })
}

/// Try to resolve path to local documentation via intra-doc-links (i.e. `super::gateway::Shard`).
///
/// See [RFC1946](https://github.com/rust-lang/rfcs/blob/master/text/1946-intra-rustdoc-links.md).
fn try_resolve_intra<T: Resolvable, D: DefDatabase + HirDatabase>(
    db: &D,
    definition: &T,
    link_text: &str,
    link_target: &str,
) -> Option<(String, String)> {
    // Set link_target for implied shortlinks
    let link_target =
        if link_target.is_empty() { link_text.trim_matches('`') } else { link_target };

    // Namespace disambiguation
    let namespace = Namespace::from_intra_spec(link_target);

    // Strip prefixes/suffixes
    let link_target = strip_prefixes_suffixes(link_target);

    // Parse link as a module path
    let path = Path::parse(link_target).ok()?;
    let modpath = ModPath::from_src(path, &Hygiene::new_unhygienic()).unwrap();

    // Resolve it relative to symbol's location (according to the RFC this should consider small scopes)
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

    Some((
        get_doc_url(db, &krate)?
            .join(&format!("{}/", krate.display_name(db)?))
            .ok()?
            .join(&path.segments.iter().map(|name| name.to_string()).join("/"))
            .ok()?
            .join(&get_symbol_filename(db, &def)?)
            .ok()?
            .into_string(),
        strip_prefixes_suffixes(link_text).to_string(),
    ))
}

/// Try to resolve path to local documentation via path-based links (i.e. `../gateway/struct.Shard.html`).
fn try_resolve_path(db: &dyn HirDatabase, moddef: &ModuleDef, link_target: &str) -> Option<String> {
    if !link_target.contains("#") && !link_target.contains(".html") {
        return None;
    }
    let ns = ItemInNs::Types(moddef.clone().into());

    let module = moddef.module(db)?;
    let krate = module.krate();
    let import_map = db.import_map(krate.into());
    let base = once(format!("{}", krate.display_name(db)?))
        .chain(import_map.path_of(ns)?.segments.iter().map(|name| format!("{}", name)))
        .join("/");

    get_doc_url(db, &krate)
        .and_then(|url| url.join(&base).ok())
        .and_then(|url| {
            get_symbol_filename(db, moddef).as_deref().map(|f| url.join(f).ok()).flatten()
        })
        .and_then(|url| url.join(link_target).ok())
        .map(|url| url.into_string())
}

// Strip prefixes, suffixes, and inline code marks from the given string.
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
    let s = s.trim_start_matches("@").trim();
    s
}

fn get_doc_url(db: &dyn HirDatabase, krate: &Crate) -> Option<Url> {
    krate
        .get_doc_url(db)
        .or_else(||
        // Fallback to docs.rs
        // FIXME: Specify an exact version here. This may be difficult, as multiple versions of the same crate could exist.
        Some(format!("https://docs.rs/{}/*/", krate.display_name(db)?)))
        .and_then(|s| Url::parse(&s).ok())
}

/// Get the filename and extension generated for a symbol by rustdoc.
///
/// Example: `struct.Shard.html`
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

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
enum Namespace {
    Types,
    Values,
    Macros,
}

static TYPES: ([&str; 7], [&str; 0]) =
    (["type", "struct", "enum", "mod", "trait", "union", "module"], []);
static VALUES: ([&str; 8], [&str; 1]) =
    (["value", "function", "fn", "method", "const", "static", "mod", "module"], ["()"]);
static MACROS: ([&str; 1], [&str; 1]) = (["macro"], ["!"]);

impl Namespace {
    /// Extract the specified namespace from an intra-doc-link if one exists.
    ///
    /// # Examples
    ///
    /// * `struct MyStruct` -> `Namespace::Types`
    /// * `panic!` -> `Namespace::Macros`
    /// * `fn@from_intra_spec` -> `Namespace::Values`
    fn from_intra_spec(s: &str) -> Option<Self> {
        [
            (Namespace::Types, (TYPES.0.iter(), TYPES.1.iter())),
            (Namespace::Values, (VALUES.0.iter(), VALUES.1.iter())),
            (Namespace::Macros, (MACROS.0.iter(), MACROS.1.iter())),
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
}

/// Sealed trait used solely for the generic bound on [`resolve_doc_link`].
pub trait Resolvable {
    fn resolver<D: DefDatabase + HirDatabase>(&self, db: &D) -> Option<Resolver>;
    fn try_into_module_def(self) -> Option<ModuleDef>;
}
