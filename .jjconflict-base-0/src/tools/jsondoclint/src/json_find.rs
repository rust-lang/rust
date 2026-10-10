use std::fmt::Write;

use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum SelectorPart {
    Field(String),
    Index(usize),
}

pub type Selector = Vec<SelectorPart>;

pub fn to_jsonpath(sel: &Selector) -> String {
    let mut s = String::from("$");
    for part in sel {
        match part {
            SelectorPart::Field(name) => {
                if is_jsonpath_safe(name) {
                    write!(&mut s, ".{}", name).unwrap();
                } else {
                    // This is probably wrong in edge cases, but all Id's are
                    // just ascii alphanumerics, `-` `_`, and `:`
                    write!(&mut s, "[{name:?}]").unwrap();
                }
            }
            SelectorPart::Index(idx) => write!(&mut s, "[{idx}]").unwrap(),
        }
    }
    s
}

fn is_jsonpath_safe(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub fn find_selector(haystack: &Value, needle: &Value) -> Vec<Selector> {
    let mut result = Vec::new();
    let mut sel = Selector::new();
    find_selector_recursive(haystack, needle, &mut result, &mut sel);
    result
}

fn find_selector_recursive(
    haystack: &Value,
    needle: &Value,
    result: &mut Vec<Selector>,
    pos: &mut Selector,
) {
    if needle == haystack {
        result.push(pos.clone());
        // Haystack cant both contain needle and be needle
    } else {
        match haystack {
            Value::Null => {}
            Value::Bool(_) => {}
            Value::Number(_) => {}
            Value::String(_) => {}
            Value::Array(arr) => {
                for (idx, subhaystack) in arr.iter().enumerate() {
                    pos.push(SelectorPart::Index(idx));
                    find_selector_recursive(subhaystack, needle, result, pos);
                    pos.pop().unwrap();
                }
            }
            Value::Object(obj) => {
                for (key, subhaystack) in obj {
                    pos.push(SelectorPart::Field(key.clone()));
                    find_selector_recursive(subhaystack, needle, result, pos);
                    pos.pop().unwrap();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
