use serde::{Deserialize, Deserializer};
use serde_derive::Deserialize;

/// This enum is used for deserializing change IDs from TOML, allowing both numeric values and the string `"ignore"`.
#[derive(Clone, Debug, PartialEq)]
pub enum ChangeId {
    Ignore,
    Id(usize),
}

/// Since we use `#[serde(deny_unknown_fields)]` on `TomlConfig`, we need a wrapper type
/// for the "change-id" field to parse it even if other fields are invalid. This ensures
/// that if deserialization fails due to other fields, we can still provide the changelogs
/// to allow developers to potentially find the reason for the failure in the logs..
#[derive(Deserialize, Default)]
pub(crate) struct ChangeIdWrapper {
    #[serde(alias = "change-id", default, deserialize_with = "deserialize_change_id")]
    pub(crate) inner: Option<ChangeId>,
}

fn deserialize_change_id<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<ChangeId>, D::Error> {
    let value = toml::Value::deserialize(deserializer)?;
    Ok(match value {
        toml::Value::String(s) if s == "ignore" => Some(ChangeId::Ignore),
        toml::Value::Integer(i) => Some(ChangeId::Id(i as usize)),
        _ => {
            return Err(serde::de::Error::custom(
                "expected \"ignore\" or an integer for change-id",
            ));
        }
    })
}
