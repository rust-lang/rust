use itertools::Itertools as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Borrow;
use std::fmt;

/// Prerenedered json.
///
/// Arrays are sorted by their stringified entries, and objects are sorted by their stringified
/// keys.
///
/// Must use serde_json with the preserve_order feature.
///
/// Both the Display and serde_json::to_string implementations write the serialized json
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(from = "Value")]
#[serde(into = "Value")]
pub(crate) struct SortedJson(String);

impl SortedJson {
    /// If you pass in an array, it will not be sorted.
    pub(crate) fn serialize<T: Serialize>(item: T) -> Self {
        SortedJson(serde_json::to_string(&item).unwrap())
    }

    /// Serializes and sorts
    pub(crate) fn array<T: Borrow<SortedJson>, I: IntoIterator<Item = T>>(items: I) -> Self {
        let items = items
            .into_iter()
            .sorted_unstable_by(|a, b| a.borrow().cmp(&b.borrow()))
            .format_with(",", |item, f| f(item.borrow()));
        SortedJson(format!("[{}]", items))
    }

    pub(crate) fn array_unsorted<T: Borrow<SortedJson>, I: IntoIterator<Item = T>>(
        items: I,
    ) -> Self {
        let items = items.into_iter().format_with(",", |item, f| f(item.borrow()));
        SortedJson(format!("[{items}]"))
    }
}

impl fmt::Display for SortedJson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Value> for SortedJson {
    fn from(value: Value) -> Self {
        SortedJson(serde_json::to_string(&value).unwrap())
    }
}

impl From<SortedJson> for Value {
    fn from(json: SortedJson) -> Self {
        serde_json::from_str(&json.0).unwrap()
    }
}

/// For use in JSON.parse('{...}').
///
/// JSON.parse supposedly loads faster than raw JS source,
/// so this is used for large objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EscapedJson(SortedJson);

impl From<SortedJson> for EscapedJson {
    fn from(json: SortedJson) -> Self {
        EscapedJson(json)
    }
}

impl fmt::Display for EscapedJson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // All these `replace` calls are because we have to go through JS string
        // for JSON content.
        // We need to escape double quotes for the JSON
        let json = self.0.0.replace('\\', r"\\").replace('\'', r"\'").replace("\\\"", "\\\\\"");
        write!(f, "{}", json)
    }
}

#[cfg(test)]
mod tests;
