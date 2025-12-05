use std::borrow::Borrow;
use std::fmt;

use itertools::Itertools as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Prerendered json.
///
/// Both the Display and serde_json::to_string implementations write the serialized json
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(from = "Value")]
#[serde(into = "Value")]
pub(crate) struct OrderedJson(String);

impl OrderedJson {
    /// If you pass in an array, it will not be sorted.
    pub(crate) fn serialize<T: Serialize>(item: T) -> Result<Self, serde_json::Error> {
        Ok(Self(serde_json::to_string(&item)?))
    }

    /// Serializes and sorts
    pub(crate) fn array_sorted<T: Borrow<Self>, I: IntoIterator<Item = T>>(items: I) -> Self {
        let items = items
            .into_iter()
            .sorted_unstable_by(|a, b| a.borrow().cmp(b.borrow()))
            .format_with(",", |item, f| f(item.borrow()));
        Self(format!("[{items}]"))
    }

    pub(crate) fn array_unsorted<T: Borrow<Self>, I: IntoIterator<Item = T>>(items: I) -> Self {
        let items = items.into_iter().format_with(",", |item, f| f(item.borrow()));
        Self(format!("[{items}]"))
    }
}

impl fmt::Display for OrderedJson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<Value> for OrderedJson {
    fn from(value: Value) -> Self {
        let serialized =
            serde_json::to_string(&value).expect("Serializing a Value to String should never fail");
        Self(serialized)
    }
}

impl From<OrderedJson> for Value {
    fn from(json: OrderedJson) -> Self {
        serde_json::from_str(&json.0).expect("OrderedJson should always store valid JSON")
    }
}

/// For use in JSON.parse('{...}').
///
/// Assumes we are going to be wrapped in single quoted strings.
///
/// JSON.parse loads faster than raw JS source,
/// so this is used for large objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EscapedJson(OrderedJson);

impl From<OrderedJson> for EscapedJson {
    fn from(json: OrderedJson) -> Self {
        Self(json)
    }
}

impl fmt::Display for EscapedJson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // All these `replace` calls are because we have to go through JS string
        // for JSON content.
        // We need to escape double quotes for the JSON
        let json = self.0.0.replace('\\', r"\\").replace('\'', r"\'").replace("\\\"", "\\\\\"");
        json.fmt(f)
    }
}

#[cfg(test)]
mod tests;
