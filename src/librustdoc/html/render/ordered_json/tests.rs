use super::super::ordered_json::*;

fn check(json: OrderedJson, serialized: &str) {
    assert_eq!(json.to_string(), serialized);
    assert_eq!(serde_json::to_string(&json).unwrap(), serialized);

    let json = json.to_string();
    let json: OrderedJson = serde_json::from_str(&json).unwrap();

    assert_eq!(json.to_string(), serialized);
    assert_eq!(serde_json::to_string(&json).unwrap(), serialized);

    let json = serde_json::to_string(&json).unwrap();
    let json: OrderedJson = serde_json::from_str(&json).unwrap();

    assert_eq!(json.to_string(), serialized);
    assert_eq!(serde_json::to_string(&json).unwrap(), serialized);
}

// Make sure there is no extra level of string, plus number of escapes.
#[test]
fn escape_json_number() {
    let json = OrderedJson::serialize(3).unwrap();
    let json = EscapedJson::from(json);
    assert_eq!(format!("{json}"), "3");
}

#[test]
fn escape_json_single_quote() {
    let json = OrderedJson::serialize("he's").unwrap();
    let json = EscapedJson::from(json);
    assert_eq!(format!("{json}"), r#""he\'s""#);
}

#[test]
fn escape_json_array() {
    let json = OrderedJson::serialize([1, 2, 3]).unwrap();
    let json = EscapedJson::from(json);
    assert_eq!(format!("{json}"), r#"[1,2,3]"#);
}

#[test]
fn escape_json_string() {
    let json = OrderedJson::serialize(r#"he"llo"#).unwrap();
    let json = EscapedJson::from(json);
    assert_eq!(format!("{json}"), r#""he\\\"llo""#);
}

#[test]
fn escape_json_string_escaped() {
    let json = OrderedJson::serialize(r#"he\"llo"#).unwrap();
    let json = EscapedJson::from(json);
    assert_eq!(format!("{json}"), r#""he\\\\\\\"llo""#);
}

#[test]
fn escape_json_string_escaped_escaped() {
    let json = OrderedJson::serialize(r#"he\\"llo"#).unwrap();
    let json = EscapedJson::from(json);
    assert_eq!(format!("{json}"), r#""he\\\\\\\\\\\"llo""#);
}

// Testing round trip + making sure there is no extra level of string
#[test]
fn number() {
    let json = OrderedJson::serialize(3).unwrap();
    let serialized = "3";
    check(json, serialized);
}

#[test]
fn boolean() {
    let json = OrderedJson::serialize(true).unwrap();
    let serialized = "true";
    check(json, serialized);
}

#[test]
fn string() {
    let json = OrderedJson::serialize("he\"llo").unwrap();
    let serialized = r#""he\"llo""#;
    check(json, serialized);
}

#[test]
fn serialize_array() {
    let json = OrderedJson::serialize([3, 1, 2]).unwrap();
    let serialized = "[3,1,2]";
    check(json, serialized);
}

#[test]
fn sorted_array() {
    let items = ["c", "a", "b"];
    let serialized = r#"["a","b","c"]"#;
    let items: Vec<OrderedJson> =
        items.into_iter().map(OrderedJson::serialize).collect::<Result<Vec<_>, _>>().unwrap();
    let json = OrderedJson::array_sorted(items);
    check(json, serialized);
}

#[test]
fn nested_array() {
    let a = OrderedJson::serialize(3).unwrap();
    let b = OrderedJson::serialize(2).unwrap();
    let c = OrderedJson::serialize(1).unwrap();
    let d = OrderedJson::serialize([1, 3, 2]).unwrap();
    let json = OrderedJson::array_sorted([a, b, c, d]);
    let serialized = r#"[1,2,3,[1,3,2]]"#;
    check(json, serialized);
}

#[test]
fn array_unsorted() {
    let items = ["c", "a", "b"];
    let serialized = r#"["c","a","b"]"#;
    let items: Vec<OrderedJson> =
        items.into_iter().map(OrderedJson::serialize).collect::<Result<Vec<_>, _>>().unwrap();
    let json = OrderedJson::array_unsorted(items);
    check(json, serialized);
}
