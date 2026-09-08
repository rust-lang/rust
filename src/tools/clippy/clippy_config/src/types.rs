use crate::de::{
    Deserialize, DeserializeOrDefault, DiagCtxt, FromDefault, TomlValue, create_value_list_msg, find_closest_match,
};
use clippy_utils::paths::{PathNS, find_crates, lookup_path};
use core::fmt::{self, Display};
use itertools::Itertools as _;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{Applicability, Diag};
use rustc_hir::PrimTy;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefIdMap;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_span::{Span, Spanned, Symbol};
use std::collections::HashMap;

macro_rules! concat_expr {
    ($($e:expr)*) => {
        concat!($($e),*)
    }
}

macro_rules! name_or_lit {
    ($name:ident) => {
        stringify!($name)
    };
    ($name:ident $lit:literal) => {
        $lit
    };
}

macro_rules! conf_enum {
    (
        $(#[$attrs:meta])*
        $vis:vis $name:ident {$(
            $(#[$var_attrs:meta])*
            $var_name:ident $(($var_lit:literal))?,
        )*}
    ) => {
        $(#[$attrs])*
        #[derive(Clone, Copy)]
        $vis enum $name {$(
            $(#[$var_attrs])*
            $var_name,
        )*}
        impl $name {
            const NAMES: &[&'static str] = &[$(name_or_lit!($var_name $($var_lit)?)),*];
            #[allow(dead_code)]
            const COUNT: usize = {
                enum __ITEMS__ { $($var_name,)* __COUNT__ }
                __ITEMS__::__COUNT__ as usize
            };

            pub fn name(self) -> &'static str {
                Self::NAMES[self as usize]
            }
            #[allow(clippy::should_implement_trait)]
            pub fn from_str(s: &str) -> Option<Self> {
                match s {
                    $(name_or_lit!($var_name $($var_lit)?) => Some(Self::$var_name),)*
                    _ => None,
                }
            }
        }
        impl FromDefault<$name> for $name {
            fn from_default(default: $name) -> Self {
                default
            }
            fn display_default(default: $name) -> impl Display {
                String::display_default(default.name())
            }
        }
        impl Deserialize for $name {
            fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
                let Some(s) = value.get_ref().as_str() else {
                    dcx.span_err(value.span(), "expected a string");
                    return None;
                };
                let x = Self::from_str(s);
                if x.is_none() {
                    let sp = dcx.make_sp(value.span());
                    let mut diag = dcx.inner.struct_span_err(
                        sp,
                        concat_expr!("expected one of: " $("`" name_or_lit!($var_name $($var_lit)?) "`")", "*),
                    );
                    if let Some(sugg) = find_closest_match(s, Self::NAMES) {
                        diag.span_suggestion(sp, "did you mean", sugg, Applicability::MaybeIncorrect);
                    }
                    diag.note(create_value_list_msg(dcx, Self::NAMES));
                    diag.emit();
                }
                x
            }
        }
    };
}
pub struct Rename {
    pub path: String,
    pub rename: String,
}

impl Deserialize for Rename {
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        if let Some(table) = value.as_ref().as_table() {
            deserialize_table!(dcx, table,
                path("path"): String,
                rename("rename"): String,
            );
            let Some(path) = path else {
                dcx.span_err(value.span().clone(), "missing required field `path`");
                return None;
            };
            let Some(rename) = rename else {
                dcx.span_err(value.span().clone(), "missing required field `rename`");
                return None;
            };
            Some(Rename { path, rename })
        } else {
            dcx.span_err(value.span(), "expected a table");
            None
        }
    }
}

pub type DisallowedPathWithoutReplacement = DisallowedPath<false>;

pub struct DisallowedPath<const REPLACEMENT_ALLOWED: bool = true> {
    path: Spanned<String>,
    reason: Option<String>,
    replacement: Option<String>,
    /// Setting `allow_invalid` to true suppresses a warning if `path` does not refer to an existing
    /// definition.
    ///
    /// This could be useful when conditional compilation is used, or when a clippy.toml file is
    /// shared among multiple projects.
    allow_invalid: bool,
}

impl<const REPLACEMENT_ALLOWED: bool> DisallowedPath<REPLACEMENT_ALLOWED> {
    pub fn path(&self) -> &str {
        &self.path.node
    }

    pub fn diag_amendment(&self, span: Span) -> impl FnOnce(&mut Diag<'_, ()>) {
        move |diag| {
            if let Some(replacement) = &self.replacement {
                diag.span_suggestion(
                    span,
                    self.reason.as_ref().map_or_else(|| String::from("use"), Clone::clone),
                    replacement,
                    Applicability::MachineApplicable,
                );
            } else if let Some(reason) = &self.reason {
                diag.note(reason.clone());
            }
        }
    }
}

impl Deserialize for DisallowedPath<false> {
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        if let Some(s) = value.as_ref().as_str() {
            Some(DisallowedPath {
                path: Spanned {
                    node: s.into(),
                    span: dcx.make_sp(value.span()),
                },
                reason: None,
                replacement: None,
                allow_invalid: false,
            })
        } else if let Some(table) = value.as_ref().as_table() {
            deserialize_table!(dcx, table,
                path("path"): Spanned<String>,
                reason("reason"): String,
                allow_invalid("allow-invalid"): bool,
            );
            let Some(path) = path else {
                dcx.span_err(value.span(), "missing required field `path`");
                return None;
            };
            Some(DisallowedPath {
                path,
                reason,
                replacement: None,
                allow_invalid: allow_invalid.unwrap_or(false),
            })
        } else {
            dcx.span_err(value.span(), "expected either a string or an inline table");
            None
        }
    }
}

impl Deserialize for DisallowedPath<true> {
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        if let Some(s) = value.as_ref().as_str() {
            Some(DisallowedPath {
                path: Spanned {
                    node: s.into(),
                    span: dcx.make_sp(value.span()),
                },
                reason: None,
                replacement: None,
                allow_invalid: false,
            })
        } else if let Some(table) = value.as_ref().as_table() {
            deserialize_table!(dcx, table,
                path("path"): Spanned<String>,
                reason("reason"): String,
                replacement("replacement"): String,
                allow_invalid("allow-invalid"): bool,
            );
            let Some(path) = path else {
                dcx.span_err(value.span(), "missing required field `path`");
                return None;
            };
            Some(DisallowedPath {
                path,
                reason,
                replacement,
                allow_invalid: allow_invalid.unwrap_or(false),
            })
        } else {
            dcx.span_err(value.span(), "expected either a string or an inline table");
            None
        }
    }
}

/// Creates a map of disallowed items to the reason they were disallowed.
#[expect(clippy::type_complexity)]
pub fn create_disallowed_map<const REPLACEMENT_ALLOWED: bool>(
    tcx: TyCtxt<'_>,
    disallowed_paths: &'static [DisallowedPath<REPLACEMENT_ALLOWED>],
    ns: PathNS,
    def_kind_predicate: impl Fn(DefKind) -> bool,
    predicate_description: &str,
    allow_prim_tys: bool,
) -> (
    DefIdMap<(&'static str, &'static DisallowedPath<REPLACEMENT_ALLOWED>)>,
    FxHashMap<PrimTy, (&'static str, &'static DisallowedPath<REPLACEMENT_ALLOWED>)>,
) {
    let mut def_ids: DefIdMap<(&'static str, &'static DisallowedPath<REPLACEMENT_ALLOWED>)> = DefIdMap::default();
    let mut prim_tys: FxHashMap<PrimTy, (&'static str, &'static DisallowedPath<REPLACEMENT_ALLOWED>)> =
        FxHashMap::default();
    for disallowed_path in disallowed_paths {
        let path = &*disallowed_path.path.node;
        let sym_path: Vec<Symbol> = path.split("::").map(Symbol::intern).collect();
        let mut resolutions = lookup_path(tcx, ns, &sym_path);
        resolutions.retain(|&def_id| def_kind_predicate(tcx.def_kind(def_id)));

        let (prim_ty, found_prim_ty) = if let &[name] = sym_path.as_slice()
            && let Some(prim) = PrimTy::from_name(name)
        {
            (allow_prim_tys.then_some(prim), true)
        } else {
            (None, false)
        };

        if resolutions.is_empty()
            && prim_ty.is_none()
            && !disallowed_path.allow_invalid
            // Don't warn about unloaded crates:
            // https://github.com/rust-lang/rust-clippy/pull/14397#issuecomment-2848328221
            && (sym_path.len() < 2 || !find_crates(tcx, sym_path[0]).is_empty())
        {
            // Relookup the path in an arbitrary namespace to get a good `expected, found` message
            let found_def_ids = lookup_path(tcx, PathNS::Arbitrary, &sym_path);
            let message = if let Some(&def_id) = found_def_ids.first() {
                let (article, description) = tcx.article_and_description(def_id);
                format!("expected a {predicate_description}, found {article} {description}")
            } else if found_prim_ty {
                format!("expected a {predicate_description}, found a primitive type")
            } else {
                format!("`{path}` does not refer to a reachable {predicate_description}")
            };
            tcx.sess
                .dcx()
                .struct_span_warn(disallowed_path.path.span, message)
                .with_help("add `allow-invalid = true` to the entry to suppress this warning")
                .emit();
        }

        for def_id in resolutions {
            def_ids.insert(def_id, (path, disallowed_path));
        }
        if let Some(ty) = prim_ty {
            prim_tys.insert(ty, (path, disallowed_path));
        }
    }

    (def_ids, prim_tys)
}

conf_enum! {
    #[derive(PartialEq, Eq)]
    pub MatchLintBehaviour {
        AllTypes,
        WellKnownTypes,
        Never,
    }
}

enum BraceKind {
    Brace,
    Bracket,
    Paren,
}

impl Deserialize for BraceKind {
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        let msg = if let Some(s) = value.as_ref().as_str() {
            match s {
                "{" | "{}" => return Some(BraceKind::Brace),
                "[" | "[]" => return Some(BraceKind::Bracket),
                "(" | "()" => return Some(BraceKind::Paren),
                _ => "unknown value",
            }
        } else {
            "expected a string"
        };
        let mut diag = dcx.inner.struct_span_err(dcx.make_sp(value.span()), msg);
        diag.note("possible values: `()`, `[]`, `{}`");
        diag.emit();
        None
    }
}

pub struct MacroMatcher {
    pub name: String,
    pub braces: (char, char),
}

impl Deserialize for MacroMatcher {
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        if let Some(table) = value.as_ref().as_table() {
            deserialize_table!(dcx, table,
                name("name"): String,
                brace("brace"): BraceKind,
            );
            let Some(name) = name else {
                dcx.span_err(value.span(), "missing required field `name`");
                return None;
            };
            let Some(brace) = brace else {
                dcx.span_err(value.span(), "missing required field `brace`");
                return None;
            };
            Some(MacroMatcher {
                name,
                braces: match brace {
                    BraceKind::Brace => ('{', '}'),
                    BraceKind::Bracket => ('[', ']'),
                    BraceKind::Paren => ('(', ')'),
                },
            })
        } else {
            dcx.span_err(value.span(), "expected an inline table");
            None
        }
    }
}

conf_enum! {
    pub PubUnderscoreFieldsBehaviour {
        PubliclyExported,
        AllPubFields,
    }
}

conf_enum! {
    /// Represents the item categories that can be ordered by the source ordering lint.
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub SourceItemOrderingCategory {
        Enum("enum"),
        Impl("impl"),
        Module("module"),
        Struct("struct"),
        Trait("trait"),
    }
}

/// Represents which item categories are enabled for ordering.
///
/// The [`Deserialize`] implementation checks that there are no duplicates in
/// the user configuration.
pub struct SourceItemOrdering(Vec<SourceItemOrderingCategory>);

impl SourceItemOrdering {
    pub fn contains(&self, category: SourceItemOrderingCategory) -> bool {
        self.0.contains(&category)
    }
}

impl fmt::Debug for SourceItemOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Deserialize for SourceItemOrdering {
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        let items = Vec::<SourceItemOrderingCategory>::deserialize(dcx, value)?;
        let mut items_set = FxHashSet::default();

        for item in &items {
            if items_set.contains(item) {
                dcx.span_err(
                    value.span(),
                    format!(
                        "The category \"{}\" was enabled more than once in the source ordering configuration.",
                        item.name()
                    ),
                );
                return None;
            }
            items_set.insert(item);
        }
        Some(SourceItemOrdering(items))
    }
}
impl FromDefault<()> for SourceItemOrdering {
    fn from_default((): ()) -> Self {
        Self(vec![
            SourceItemOrderingCategory::Enum,
            SourceItemOrderingCategory::Impl,
            SourceItemOrderingCategory::Module,
            SourceItemOrderingCategory::Struct,
            SourceItemOrderingCategory::Trait,
        ])
    }
    fn display_default((): ()) -> impl Display {
        r#"["enum", "impl", "module", "struct", "trait"]"#
    }
}
impl DeserializeOrDefault<()> for SourceItemOrdering {
    fn deserialize_or_default(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>, default: ()) -> Self {
        Self::deserialize(dcx, value).unwrap_or_else(|| Self::from_default(default))
    }
}

conf_enum! {
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub SourceItemOrderingModuleItemKind {
        ExternCrate("extern_crate"),
        Mod("mod"),
        ForeignMod("foreign_mod"),
        Use("use"),
        Macro("macro"),
        GlobalAsm("global_asm"),
        Static("static"),
        Const("const"),
        TyAlias("ty_alias"),
        Enum("enum"),
        Struct("struct"),
        Union("union"),
        Trait("trait"),
        TraitAlias("trait_alias"),
        Impl("impl"),
        Fn("fn"),
        TestBinderConstraints("test_binder_constraints"),
    }
}

impl SourceItemOrderingModuleItemKind {
    pub fn all_variants() -> Vec<Self> {
        #[allow(clippy::enum_glob_use)] // Very local glob use for legibility.
        use SourceItemOrderingModuleItemKind::*;
        vec![
            ExternCrate,
            Mod,
            ForeignMod,
            Use,
            Macro,
            GlobalAsm,
            Static,
            Const,
            TyAlias,
            Enum,
            Struct,
            Union,
            Trait,
            TraitAlias,
            Impl,
            Fn,
            TestBinderConstraints,
        ]
    }
}

/// Represents the configured ordering of items within a module.
///
/// The [`Deserialize`] implementation checks that no item kinds have been
/// omitted and that there are no duplicates in the user configuration.
#[derive(Clone)]
pub struct SourceItemOrderingModuleItemGroupings {
    groups: Vec<(String, Vec<SourceItemOrderingModuleItemKind>)>,
    lut: HashMap<SourceItemOrderingModuleItemKind, usize>,
    back_lut: HashMap<SourceItemOrderingModuleItemKind, String>,
}

impl fmt::Debug for SourceItemOrderingModuleItemGroupings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.groups.fmt(f)
    }
}

impl SourceItemOrderingModuleItemGroupings {
    fn build_lut(
        groups: &[(String, Vec<SourceItemOrderingModuleItemKind>)],
    ) -> HashMap<SourceItemOrderingModuleItemKind, usize> {
        let mut lut = HashMap::new();
        for (group_index, (_, items)) in groups.iter().enumerate() {
            for &item in items {
                lut.insert(item, group_index);
            }
        }
        lut
    }

    fn build_back_lut(
        groups: &[(String, Vec<SourceItemOrderingModuleItemKind>)],
    ) -> HashMap<SourceItemOrderingModuleItemKind, String> {
        let mut lut = HashMap::new();
        for (group_name, items) in groups {
            for &item in items {
                lut.insert(item, group_name.clone());
            }
        }
        lut
    }

    pub fn grouping_name_of(&self, item: SourceItemOrderingModuleItemKind) -> Option<&String> {
        self.back_lut.get(&item)
    }

    pub fn grouping_names(&self) -> Vec<String> {
        self.groups.iter().map(|(name, _)| name.clone()).collect()
    }

    pub fn is_grouping(&self, grouping: &str) -> bool {
        self.groups.iter().any(|(g, _)| g == grouping)
    }

    pub fn module_level_order_of(&self, item: SourceItemOrderingModuleItemKind) -> Option<usize> {
        self.lut.get(&item).copied()
    }
}

impl Deserialize for SourceItemOrderingModuleItemGroupings {
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        let Some(values) = value.as_ref().as_array() else {
            dcx.span_err(value.span(), "expected an array");
            return None;
        };
        let mut groups = Vec::with_capacity(values.len());
        for value in values {
            if let Some(values) = value.as_ref().as_array()
                && let [value1, value2] = &**values
            {
                groups.push((
                    String::deserialize(dcx, value1)?,
                    Vec::<SourceItemOrderingModuleItemKind>::deserialize(dcx, value2)?,
                ));
            } else {
                dcx.span_err(value.span(), "expected an array of length two");
                return None;
            }
        }

        let items_total: usize = groups.iter().map(|(_, v)| v.len()).sum();
        let lut = Self::build_lut(&groups);
        let back_lut = Self::build_back_lut(&groups);

        let mut expected_items = SourceItemOrderingModuleItemKind::all_variants();
        for item in lut.keys() {
            expected_items.retain(|i| i != item);
        }

        let all_items = SourceItemOrderingModuleItemKind::all_variants();
        if expected_items.is_empty() && items_total == all_items.len() {
            let Some(use_group_index) = lut.get(&SourceItemOrderingModuleItemKind::Use) else {
                dcx.span_err(value.span(), "Error in internal LUT.");
                return None;
            };
            let Some((_, use_group_items)) = groups.get(*use_group_index) else {
                dcx.span_err(value.span(), "Error in internal LUT.");
                return None;
            };
            if use_group_items.len() > 1 {
                dcx.span_err(
                    value.span(),
                    "The group containing the \"use\" item kind may not contain any other item kinds. \
                    The \"use\" items will (generally) be sorted by rustfmt already. \
                    Therefore it makes no sense to implement linting rules that may conflict with rustfmt.",
                );
                return None;
            }
            Some(Self { groups, lut, back_lut })
        } else if items_total != all_items.len() {
            dcx.span_err(value.span(),
                format!(
                    "Some module item kinds were configured more than once, or were missing, in the source ordering configuration. \
                    The module item kinds are: {all_items:?}"
                )
            );
            None
        } else {
            dcx.span_err(value.span(),
                format!(
                    "Not all module item kinds were part of the configured source ordering rule. \
                    All item kinds must be provided in the config, otherwise the required source ordering would remain ambiguous. \
                    The module item kinds are: {all_items:?}"
                )
            );
            None
        }
    }
}
impl FromDefault<()> for SourceItemOrderingModuleItemGroupings {
    fn from_default((): ()) -> Self {
        Self {
            groups: vec![
                (
                    "modules".into(),
                    vec![
                        SourceItemOrderingModuleItemKind::ExternCrate,
                        SourceItemOrderingModuleItemKind::Mod,
                        SourceItemOrderingModuleItemKind::ForeignMod,
                    ],
                ),
                ("use".into(), vec![SourceItemOrderingModuleItemKind::Use]),
                ("macros".into(), vec![SourceItemOrderingModuleItemKind::Macro]),
                ("global_asm".into(), vec![SourceItemOrderingModuleItemKind::GlobalAsm]),
                (
                    "UPPER_SNAKE_CASE".into(),
                    vec![
                        SourceItemOrderingModuleItemKind::Static,
                        SourceItemOrderingModuleItemKind::Const,
                    ],
                ),
                (
                    "PascalCase".into(),
                    vec![
                        SourceItemOrderingModuleItemKind::TyAlias,
                        SourceItemOrderingModuleItemKind::Enum,
                        SourceItemOrderingModuleItemKind::Struct,
                        SourceItemOrderingModuleItemKind::Union,
                        SourceItemOrderingModuleItemKind::Trait,
                        SourceItemOrderingModuleItemKind::TraitAlias,
                        SourceItemOrderingModuleItemKind::Impl,
                        SourceItemOrderingModuleItemKind::TestBinderConstraints,
                    ],
                ),
                ("lower_snake_case".into(), vec![SourceItemOrderingModuleItemKind::Fn]),
            ],
            lut: HashMap::from_iter([
                (SourceItemOrderingModuleItemKind::ExternCrate, 0),
                (SourceItemOrderingModuleItemKind::Mod, 0),
                (SourceItemOrderingModuleItemKind::ForeignMod, 0),
                (SourceItemOrderingModuleItemKind::Use, 1),
                (SourceItemOrderingModuleItemKind::Macro, 2),
                (SourceItemOrderingModuleItemKind::GlobalAsm, 3),
                (SourceItemOrderingModuleItemKind::Static, 4),
                (SourceItemOrderingModuleItemKind::Const, 4),
                (SourceItemOrderingModuleItemKind::TyAlias, 5),
                (SourceItemOrderingModuleItemKind::Enum, 5),
                (SourceItemOrderingModuleItemKind::Struct, 5),
                (SourceItemOrderingModuleItemKind::Union, 5),
                (SourceItemOrderingModuleItemKind::Trait, 5),
                (SourceItemOrderingModuleItemKind::TraitAlias, 5),
                (SourceItemOrderingModuleItemKind::Impl, 5),
                (SourceItemOrderingModuleItemKind::TestBinderConstraints, 5),
                (SourceItemOrderingModuleItemKind::Fn, 6),
            ]),
            back_lut: HashMap::from_iter([
                (SourceItemOrderingModuleItemKind::ExternCrate, "modules".into()),
                (SourceItemOrderingModuleItemKind::Mod, "modules".into()),
                (SourceItemOrderingModuleItemKind::ForeignMod, "modules".into()),
                (SourceItemOrderingModuleItemKind::Use, "use".into()),
                (SourceItemOrderingModuleItemKind::Macro, "macros".into()),
                (SourceItemOrderingModuleItemKind::GlobalAsm, "global_asm".into()),
                (SourceItemOrderingModuleItemKind::Static, "UPPER_SNAKE_CASE".into()),
                (SourceItemOrderingModuleItemKind::Const, "UPPER_SNAKE_CASE".into()),
                (SourceItemOrderingModuleItemKind::TyAlias, "PascalCase".into()),
                (SourceItemOrderingModuleItemKind::Enum, "PascalCase".into()),
                (SourceItemOrderingModuleItemKind::Struct, "PascalCase".into()),
                (SourceItemOrderingModuleItemKind::Union, "PascalCase".into()),
                (SourceItemOrderingModuleItemKind::Trait, "PascalCase".into()),
                (SourceItemOrderingModuleItemKind::TraitAlias, "PascalCase".into()),
                (SourceItemOrderingModuleItemKind::Impl, "PascalCase".into()),
                (
                    SourceItemOrderingModuleItemKind::TestBinderConstraints,
                    "PascalCase".into(),
                ),
                (SourceItemOrderingModuleItemKind::Fn, "lower_snake_case".into()),
            ]),
        }
    }
    fn display_default((): ()) -> impl Display {
        r#"[["modules", ["extern_crate", "mod", "foreign_mod"]], ["use", ["use"]], ["macros", ["macro"]], ["global_asm", ["global_asm"]], ["UPPER_SNAKE_CASE", ["static", "const"]], ["PascalCase", ["ty_alias", "enum", "struct", "union", "trait", "trait_alias", "impl", "test_binder_constraints"]], ["lower_snake_case", ["fn"]]]"#
    }
}
impl DeserializeOrDefault<()> for SourceItemOrderingModuleItemGroupings {
    fn deserialize_or_default(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>, default: ()) -> Self {
        Self::deserialize(dcx, value).unwrap_or_else(|| Self::from_default(default))
    }
}

conf_enum! {
    #[derive(Debug, PartialEq)]
    pub SourceItemOrderingTraitAssocItemKind {
        Const("const"),
        Fn("fn"),
        Type("type"),
    }
}

impl SourceItemOrderingTraitAssocItemKind {
    pub fn all_variants() -> Vec<Self> {
        #[allow(clippy::enum_glob_use)] // Very local glob use for legibility.
        use SourceItemOrderingTraitAssocItemKind::*;
        vec![Const, Fn, Type]
    }
}

/// Represents the order in which associated trait items should be ordered.
///
/// The reason to wrap a `Vec` in a newtype is to be able to implement
/// [`Deserialize`]. Implementing `Deserialize` allows for implementing checks
/// on configuration completeness at the time of loading the clippy config,
/// letting the user know if there's any issues with the config (e.g. not
/// listing all item kinds that should be sorted).
#[derive(Clone)]
pub struct SourceItemOrderingTraitAssocItemKinds(Vec<SourceItemOrderingTraitAssocItemKind>);

impl fmt::Debug for SourceItemOrderingTraitAssocItemKinds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl SourceItemOrderingTraitAssocItemKinds {
    pub fn index_of(&self, item: SourceItemOrderingTraitAssocItemKind) -> Option<usize> {
        self.0.iter().position(|&i| i == item)
    }
}

impl Deserialize for SourceItemOrderingTraitAssocItemKinds {
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        let items = Vec::<SourceItemOrderingTraitAssocItemKind>::deserialize(dcx, value)?;

        let mut expected_items = SourceItemOrderingTraitAssocItemKind::all_variants();
        for item in &items {
            expected_items.retain(|i| i != item);
        }

        let all_items = SourceItemOrderingTraitAssocItemKind::all_variants();
        if expected_items.is_empty() && items.len() == all_items.len() {
            Some(Self(items))
        } else if items.len() != all_items.len() {
            dcx.span_err(
                value.span(),
                format!(
                    "Some trait associated item kinds were configured more than once, or were missing, in the source ordering configuration. \
                    The trait associated item kinds are: {all_items:?}",
                )
            );
            None
        } else {
            dcx.span_err(
                value.span(),
                format!(
                    "Not all trait associated item kinds were part of the configured source ordering rule. \
                    All item kinds must be provided in the config, otherwise the required source ordering would remain ambiguous. \
                    The trait associated item kinds are: {all_items:?}"
                )
            );
            None
        }
    }
}
impl FromDefault<()> for SourceItemOrderingTraitAssocItemKinds {
    fn from_default((): ()) -> Self {
        Self(vec![
            SourceItemOrderingTraitAssocItemKind::Const,
            SourceItemOrderingTraitAssocItemKind::Type,
            SourceItemOrderingTraitAssocItemKind::Fn,
        ])
    }
    fn display_default((): ()) -> impl Display {
        r#"["const", "type", "fn"]"#
    }
}
impl DeserializeOrDefault<()> for SourceItemOrderingTraitAssocItemKinds {
    fn deserialize_or_default(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>, default: ()) -> Self {
        Self::deserialize(dcx, value).unwrap_or_else(|| Self::from_default(default))
    }
}

/// Describes which specific groupings should have their items ordered
/// alphabetically.
///
/// This is separate from defining and enforcing groupings. For example,
/// defining enums are grouped before structs still allows for an enum B to be
/// placed before an enum A. Only when enforcing ordering within the grouping,
/// will it be checked if A is placed before B.
#[derive(Clone, Debug)]
pub enum SourceItemOrderingWithinModuleItemGroupings {
    /// All groupings should have their items ordered.
    All,

    /// None of the groupings should have their order checked.
    None,

    /// Only the specified groupings should have their order checked.
    Custom(Vec<Spanned<String>>),
}

impl SourceItemOrderingWithinModuleItemGroupings {
    pub fn ordered_within(&self, grouping_name: &String) -> bool {
        match self {
            SourceItemOrderingWithinModuleItemGroupings::All => true,
            SourceItemOrderingWithinModuleItemGroupings::None => false,
            SourceItemOrderingWithinModuleItemGroupings::Custom(groups) => {
                groups.iter().any(|x| x.node == *grouping_name)
            },
        }
    }

    pub fn check_groupings(&self, sess: &Session, module_item_order_groupings: &SourceItemOrderingModuleItemGroupings) {
        if let SourceItemOrderingWithinModuleItemGroupings::Custom(groupings) = self {
            for grouping in groupings {
                if !module_item_order_groupings.is_grouping(&grouping.node) {
                    // Since this isn't fixable by rustfix, don't emit a `Suggestion`. This just adds some useful
                    // info for the user instead.
                    let names = module_item_order_groupings
                        .groups
                        .iter()
                        .map(|(x, _)| &**x)
                        .collect::<Vec<_>>();
                    let suggestion = find_closest_match(&grouping.node, &names)
                        .map(|s| format!(" perhaps you meant `{s}`?"))
                        .unwrap_or_default();
                    let names = names.iter().map(|s| format!("`{s}`")).join(", ");
                    sess.dcx().span_err(grouping.span, format!(
                        "unknown ordering group: `{}` was not specified in `module-items-ordered-within-groupings`,{suggestion} expected one of: {names}",
                        grouping.node,
                    ));
                }
            }
        }
    }
}

impl Deserialize for SourceItemOrderingWithinModuleItemGroupings {
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        match value.as_ref() {
            toml::de::DeValue::String(str_value) => match &**str_value {
                "all" => Some(Self::All),
                "none" => Some(Self::None),
                _ => {
                    dcx.span_err(value.span(), "expected: `all`, `none` or a list of category names");
                    None
                },
            },
            toml::de::DeValue::Array(_) => Vec::deserialize(dcx, value).map(Self::Custom),
            _ => {
                dcx.span_err(value.span(), "expected a string or an array of strings");
                None
            },
        }
    }
}
impl FromDefault<()> for SourceItemOrderingWithinModuleItemGroupings {
    fn from_default((): ()) -> Self {
        Self::None
    }
    fn display_default((): ()) -> impl Display {
        r#""none""#
    }
}
impl DeserializeOrDefault<()> for SourceItemOrderingWithinModuleItemGroupings {
    fn deserialize_or_default(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>, default: ()) -> Self {
        Self::deserialize(dcx, value).unwrap_or_else(|| Self::from_default(default))
    }
}

conf_enum! {
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub InherentImplLintScope {
        Crate("crate"),
        File("file"),
        Module("module"),
    }
}

conf_enum! {
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub TraitImplItemOrder {
        Alphabetical("alphabetical"),
        TraitItemOrdering("trait_item_ordering"),
        AlphabeticalOrTraitItemOrdering("alphabetical_or_trait_item_ordering"),
    }
}
impl FromDefault<()> for TraitImplItemOrder {
    fn from_default((): ()) -> Self {
        Self::Alphabetical
    }
    fn display_default((): ()) -> impl Display {
        r#""alphabetical""#
    }
}
impl DeserializeOrDefault<()> for TraitImplItemOrder {
    fn deserialize_or_default(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>, default: ()) -> Self {
        Self::deserialize(dcx, value).unwrap_or_else(|| Self::from_default(default))
    }
}
