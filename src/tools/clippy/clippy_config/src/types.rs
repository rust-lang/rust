use clippy_utils::paths::{PathNS, find_crates, lookup_path};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{Applicability, Diag};
use rustc_hir::PrimTy;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefIdMap;
use rustc_middle::ty::TyCtxt;
use rustc_span::{Span, Symbol};
use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize, ser};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rename {
    pub path: String,
    pub rename: String,
}

pub type DisallowedPathWithoutReplacement = DisallowedPath<false>;

#[derive(Debug, Serialize)]
pub struct DisallowedPath<const REPLACEMENT_ALLOWED: bool = true> {
    path: String,
    reason: Option<String>,
    replacement: Option<String>,
    /// Setting `allow_invalid` to true suppresses a warning if `path` does not refer to an existing
    /// definition.
    ///
    /// This could be useful when conditional compilation is used, or when a clippy.toml file is
    /// shared among multiple projects.
    allow_invalid: bool,
    /// The span of the `DisallowedPath`.
    ///
    /// Used for diagnostics.
    #[serde(skip_serializing)]
    span: Span,
}

impl<'de, const REPLACEMENT_ALLOWED: bool> Deserialize<'de> for DisallowedPath<REPLACEMENT_ALLOWED> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let enum_ = DisallowedPathEnum::deserialize(deserializer)?;
        if !REPLACEMENT_ALLOWED && enum_.replacement().is_some() {
            return Err(de::Error::custom("replacement not allowed for this configuration"));
        }
        Ok(Self {
            path: enum_.path().to_owned(),
            reason: enum_.reason().map(ToOwned::to_owned),
            replacement: enum_.replacement().map(ToOwned::to_owned),
            allow_invalid: enum_.allow_invalid(),
            span: Span::default(),
        })
    }
}

// `DisallowedPathEnum` is an implementation detail to enable the `Deserialize` implementation just
// above. `DisallowedPathEnum` is not meant to be used outside of this file.
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged, deny_unknown_fields)]
enum DisallowedPathEnum {
    Simple(String),
    WithReason {
        path: String,
        reason: Option<String>,
        replacement: Option<String>,
        #[serde(rename = "allow-invalid")]
        allow_invalid: Option<bool>,
    },
}

impl<const REPLACEMENT_ALLOWED: bool> DisallowedPath<REPLACEMENT_ALLOWED> {
    pub fn path(&self) -> &str {
        &self.path
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

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }
}

impl DisallowedPathEnum {
    pub fn path(&self) -> &str {
        let (Self::Simple(path) | Self::WithReason { path, .. }) = self;

        path
    }

    fn reason(&self) -> Option<&str> {
        match &self {
            Self::WithReason { reason, .. } => reason.as_deref(),
            Self::Simple(_) => None,
        }
    }

    fn replacement(&self) -> Option<&str> {
        match &self {
            Self::WithReason { replacement, .. } => replacement.as_deref(),
            Self::Simple(_) => None,
        }
    }

    fn allow_invalid(&self) -> bool {
        match &self {
            Self::WithReason { allow_invalid, .. } => allow_invalid.unwrap_or_default(),
            Self::Simple(_) => false,
        }
    }
}

/// Creates a map of disallowed items to the reason they were disallowed.
#[allow(clippy::type_complexity)]
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
        let path = disallowed_path.path();
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
                .struct_span_warn(disallowed_path.span(), message)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum MatchLintBehaviour {
    AllTypes,
    WellKnownTypes,
    Never,
}

#[derive(Debug)]
pub struct MacroMatcher {
    pub name: String,
    pub braces: (char, char),
}

impl<'de> Deserialize<'de> for MacroMatcher {
    fn deserialize<D>(deser: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Name,
            Brace,
        }
        struct MacVisitor;
        impl<'de> Visitor<'de> for MacVisitor {
            type Value = MacroMatcher;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("struct MacroMatcher")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
            where
                V: de::MapAccess<'de>,
            {
                let mut name = None;
                let mut brace: Option<char> = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Name => {
                            if name.is_some() {
                                return Err(de::Error::duplicate_field("name"));
                            }
                            name = Some(map.next_value()?);
                        },
                        Field::Brace => {
                            if brace.is_some() {
                                return Err(de::Error::duplicate_field("brace"));
                            }
                            brace = Some(map.next_value()?);
                        },
                    }
                }
                let name = name.ok_or_else(|| de::Error::missing_field("name"))?;
                let brace = brace.ok_or_else(|| de::Error::missing_field("brace"))?;
                Ok(MacroMatcher {
                    name,
                    braces: [('(', ')'), ('{', '}'), ('[', ']')]
                        .into_iter()
                        .find(|b| b.0 == brace)
                        .map(|(o, c)| (o.to_owned(), c.to_owned()))
                        .ok_or_else(|| de::Error::custom(format!("expected one of `(`, `{{`, `[` found `{brace}`")))?,
                })
            }
        }

        const FIELDS: &[&str] = &["name", "brace"];
        deser.deserialize_struct("MacroMatcher", FIELDS, MacVisitor)
    }
}

/// Represents the item categories that can be ordered by the source ordering lint.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceItemOrderingCategory {
    Enum,
    Impl,
    Module,
    Struct,
    Trait,
}

/// Represents which item categories are enabled for ordering.
///
/// The [`Deserialize`] implementation checks that there are no duplicates in
/// the user configuration.
pub struct SourceItemOrdering(Vec<SourceItemOrderingCategory>);

impl SourceItemOrdering {
    pub fn contains(&self, category: &SourceItemOrderingCategory) -> bool {
        self.0.contains(category)
    }
}

impl<T> From<T> for SourceItemOrdering
where
    T: Into<Vec<SourceItemOrderingCategory>>,
{
    fn from(value: T) -> Self {
        Self(value.into())
    }
}

impl core::fmt::Debug for SourceItemOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<'de> Deserialize<'de> for SourceItemOrdering {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let items = Vec::<SourceItemOrderingCategory>::deserialize(deserializer)?;
        let mut items_set = std::collections::HashSet::new();

        for item in &items {
            if items_set.contains(item) {
                return Err(de::Error::custom(format!(
                    "The category \"{item:?}\" was enabled more than once in the source ordering configuration."
                )));
            }
            items_set.insert(item);
        }

        Ok(Self(items))
    }
}

impl Serialize for SourceItemOrdering {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: ser::Serializer,
    {
        self.0.serialize(serializer)
    }
}

/// Represents the items that can occur within a module.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceItemOrderingModuleItemKind {
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

impl SourceItemOrderingModuleItemGroupings {
    fn build_lut(
        groups: &[(String, Vec<SourceItemOrderingModuleItemKind>)],
    ) -> HashMap<SourceItemOrderingModuleItemKind, usize> {
        let mut lut = HashMap::new();
        for (group_index, (_, items)) in groups.iter().enumerate() {
            for item in items {
                lut.insert(item.clone(), group_index);
            }
        }
        lut
    }

    fn build_back_lut(
        groups: &[(String, Vec<SourceItemOrderingModuleItemKind>)],
    ) -> HashMap<SourceItemOrderingModuleItemKind, String> {
        let mut lut = HashMap::new();
        for (group_name, items) in groups {
            for item in items {
                lut.insert(item.clone(), group_name.clone());
            }
        }
        lut
    }

    pub fn grouping_name_of(&self, item: &SourceItemOrderingModuleItemKind) -> Option<&String> {
        self.back_lut.get(item)
    }

    pub fn grouping_names(&self) -> Vec<String> {
        self.groups.iter().map(|(name, _)| name.clone()).collect()
    }

    pub fn is_grouping(&self, grouping: &str) -> bool {
        self.groups.iter().any(|(g, _)| g == grouping)
    }

    pub fn module_level_order_of(&self, item: &SourceItemOrderingModuleItemKind) -> Option<usize> {
        self.lut.get(item).copied()
    }
}

impl From<&[(&str, &[SourceItemOrderingModuleItemKind])]> for SourceItemOrderingModuleItemGroupings {
    fn from(value: &[(&str, &[SourceItemOrderingModuleItemKind])]) -> Self {
        let groups: Vec<(String, Vec<SourceItemOrderingModuleItemKind>)> =
            value.iter().map(|item| (item.0.to_string(), item.1.to_vec())).collect();
        let lut = Self::build_lut(&groups);
        let back_lut = Self::build_back_lut(&groups);
        Self { groups, lut, back_lut }
    }
}

impl core::fmt::Debug for SourceItemOrderingModuleItemGroupings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.groups.fmt(f)
    }
}

impl<'de> Deserialize<'de> for SourceItemOrderingModuleItemGroupings {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let groups = Vec::<(String, Vec<SourceItemOrderingModuleItemKind>)>::deserialize(deserializer)?;
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
                return Err(de::Error::custom("Error in internal LUT."));
            };
            let Some((_, use_group_items)) = groups.get(*use_group_index) else {
                return Err(de::Error::custom("Error in internal LUT."));
            };
            if use_group_items.len() > 1 {
                return Err(de::Error::custom(
                    "The group containing the \"use\" item kind may not contain any other item kinds. \
                    The \"use\" items will (generally) be sorted by rustfmt already. \
                    Therefore it makes no sense to implement linting rules that may conflict with rustfmt.",
                ));
            }

            Ok(Self { groups, lut, back_lut })
        } else if items_total != all_items.len() {
            Err(de::Error::custom(format!(
                "Some module item kinds were configured more than once, or were missing, in the source ordering configuration. \
                The module item kinds are: {all_items:?}"
            )))
        } else {
            Err(de::Error::custom(format!(
                "Not all module item kinds were part of the configured source ordering rule. \
                All item kinds must be provided in the config, otherwise the required source ordering would remain ambiguous. \
                The module item kinds are: {all_items:?}"
            )))
        }
    }
}

impl Serialize for SourceItemOrderingModuleItemGroupings {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: ser::Serializer,
    {
        self.groups.serialize(serializer)
    }
}

/// Represents all kinds of trait associated items.
#[derive(Clone, Debug, Deserialize, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceItemOrderingTraitAssocItemKind {
    Const,
    Fn,
    Type,
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

impl SourceItemOrderingTraitAssocItemKinds {
    pub fn index_of(&self, item: &SourceItemOrderingTraitAssocItemKind) -> Option<usize> {
        self.0.iter().position(|i| i == item)
    }
}

impl<T> From<T> for SourceItemOrderingTraitAssocItemKinds
where
    T: Into<Vec<SourceItemOrderingTraitAssocItemKind>>,
{
    fn from(value: T) -> Self {
        Self(value.into())
    }
}

impl core::fmt::Debug for SourceItemOrderingTraitAssocItemKinds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<'de> Deserialize<'de> for SourceItemOrderingTraitAssocItemKinds {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let items = Vec::<SourceItemOrderingTraitAssocItemKind>::deserialize(deserializer)?;

        let mut expected_items = SourceItemOrderingTraitAssocItemKind::all_variants();
        for item in &items {
            expected_items.retain(|i| i != item);
        }

        let all_items = SourceItemOrderingTraitAssocItemKind::all_variants();
        if expected_items.is_empty() && items.len() == all_items.len() {
            Ok(Self(items))
        } else if items.len() != all_items.len() {
            Err(de::Error::custom(format!(
                "Some trait associated item kinds were configured more than once, or were missing, in the source ordering configuration. \
                The trait associated item kinds are: {all_items:?}",
            )))
        } else {
            Err(de::Error::custom(format!(
                "Not all trait associated item kinds were part of the configured source ordering rule. \
                All item kinds must be provided in the config, otherwise the required source ordering would remain ambiguous. \
                The trait associated item kinds are: {all_items:?}"
            )))
        }
    }
}

impl Serialize for SourceItemOrderingTraitAssocItemKinds {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: ser::Serializer,
    {
        self.0.serialize(serializer)
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
    Custom(Vec<String>),
}

impl SourceItemOrderingWithinModuleItemGroupings {
    pub fn ordered_within(&self, grouping_name: &String) -> bool {
        match self {
            SourceItemOrderingWithinModuleItemGroupings::All => true,
            SourceItemOrderingWithinModuleItemGroupings::None => false,
            SourceItemOrderingWithinModuleItemGroupings::Custom(groups) => groups.contains(grouping_name),
        }
    }
}

/// Helper struct for deserializing the [`SourceItemOrderingWithinModuleItemGroupings`].
#[derive(Deserialize)]
#[serde(untagged)]
enum StringOrVecOfString {
    String(String),
    Vec(Vec<String>),
}

impl<'de> Deserialize<'de> for SourceItemOrderingWithinModuleItemGroupings {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let description = "The available options for configuring an ordering within module item groups are: \
                    \"all\", \"none\", or a list of module item group names \
                    (as configured with the `module-item-order-groupings` configuration option).";

        match StringOrVecOfString::deserialize(deserializer) {
            Ok(StringOrVecOfString::String(preset)) if preset == "all" => {
                Ok(SourceItemOrderingWithinModuleItemGroupings::All)
            },
            Ok(StringOrVecOfString::String(preset)) if preset == "none" => {
                Ok(SourceItemOrderingWithinModuleItemGroupings::None)
            },
            Ok(StringOrVecOfString::String(preset)) => Err(de::Error::custom(format!(
                "Unknown configuration option: {preset}.\n{description}"
            ))),
            Ok(StringOrVecOfString::Vec(groupings)) => {
                Ok(SourceItemOrderingWithinModuleItemGroupings::Custom(groupings))
            },
            Err(e) => Err(de::Error::custom(format!("{e}\n{description}"))),
        }
    }
}

impl Serialize for SourceItemOrderingWithinModuleItemGroupings {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: ser::Serializer,
    {
        match self {
            SourceItemOrderingWithinModuleItemGroupings::All => serializer.serialize_str("all"),
            SourceItemOrderingWithinModuleItemGroupings::None => serializer.serialize_str("none"),
            SourceItemOrderingWithinModuleItemGroupings::Custom(vec) => vec.serialize(serializer),
        }
    }
}

// these impls are never actually called but are used by the various config options that default to
// empty lists
macro_rules! unimplemented_serialize {
    ($($t:ty,)*) => {
        $(
            impl Serialize for $t {
                fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: ser::Serializer,
                {
                    Err(ser::Error::custom("unimplemented"))
                }
            }
        )*
    }
}

unimplemented_serialize! {
    Rename,
    MacroMatcher,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum PubUnderscoreFieldsBehaviour {
    PubliclyExported,
    AllPubFields,
}
