use clippy_utils::def_path_def_ids;
use rustc_hir::def_id::DefIdMap;
use rustc_middle::ty::TyCtxt;
use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize, ser};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Deserialize)]
pub struct Rename {
    pub path: String,
    pub rename: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum DisallowedPath {
    Simple(String),
    WithReason { path: String, reason: Option<String> },
}

impl DisallowedPath {
    pub fn path(&self) -> &str {
        let (Self::Simple(path) | Self::WithReason { path, .. }) = self;

        path
    }

    pub fn reason(&self) -> Option<&str> {
        match &self {
            Self::WithReason { reason, .. } => reason.as_deref(),
            Self::Simple(_) => None,
        }
    }
}

/// Creates a map of disallowed items to the reason they were disallowed.
pub fn create_disallowed_map(
    tcx: TyCtxt<'_>,
    disallowed: &'static [DisallowedPath],
) -> DefIdMap<(&'static str, Option<&'static str>)> {
    disallowed
        .iter()
        .map(|x| (x.path(), x.path().split("::").collect::<Vec<_>>(), x.reason()))
        .flat_map(|(name, path, reason)| def_path_def_ids(tcx, &path).map(move |id| (id, (name, reason))))
        .collect()
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

    pub fn module_level_order_of(&self, item: &SourceItemOrderingModuleItemKind) -> Option<usize> {
        self.lut.get(item).copied()
    }
}

impl From<&[(&str, &[SourceItemOrderingModuleItemKind])]> for SourceItemOrderingModuleItemGroupings {
    fn from(value: &[(&str, &[SourceItemOrderingModuleItemKind])]) -> Self {
        let groups: Vec<(String, Vec<SourceItemOrderingModuleItemKind>)> =
            value.iter().map(|item| (item.0.to_string(), item.1.to_vec())).collect();
        let lut = Self::build_lut(&groups);
        Self { groups, lut }
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

            Ok(Self { groups, lut })
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
    DisallowedPath,
    Rename,
    MacroMatcher,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum PubUnderscoreFieldsBehaviour {
    PubliclyExported,
    AllPubFields,
}
