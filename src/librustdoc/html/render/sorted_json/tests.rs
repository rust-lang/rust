use super::super::sorted_json::*;

fn check(json: SortedJson, serialized: &str) {
    assert_eq!(json.to_string(), serialized);
    assert_eq!(serde_json::to_string(&json).unwrap(), serialized);

    let json = json.to_string();
    let json: SortedJson = serde_json::from_str(&json).unwrap();

    assert_eq!(json.to_string(), serialized);
    assert_eq!(serde_json::to_string(&json).unwrap(), serialized);

    let json = serde_json::to_string(&json).unwrap();
    let json: SortedJson = serde_json::from_str(&json).unwrap();

    assert_eq!(json.to_string(), serialized);
    assert_eq!(serde_json::to_string(&json).unwrap(), serialized);
}

// Test this basic are needed because we are testing that our Display impl + serialize impl don't
// nest everything in extra level of string. We also are testing round trip.
#[test]
fn escape_json_number() {
    let json = SortedJson::serialize(3);
    let json = EscapedJson::from(json);
    assert_eq!(format!("{json}"), "3");
}

#[test]
fn escape_json_single_quote() {
    let json = SortedJson::serialize("he's");
    let json = EscapedJson::from(json);
    assert_eq!(format!("{json}"), r#""he\'s""#);
}

#[test]
fn escape_json_array() {
    let json = SortedJson::serialize([1, 2, 3]);
    let json = EscapedJson::from(json);
    assert_eq!(format!("{json}"), r#"[1,2,3]"#);
}

#[test]
fn escape_json_string() {
    let json = SortedJson::serialize(r#"he"llo"#);
    let json = EscapedJson::from(json);
    assert_eq!(format!("{json}"), r#""he\\\"llo""#);
}

#[test]
fn escape_json_string_escaped() {
    let json = SortedJson::serialize(r#"he\"llo"#);
    let json = EscapedJson::from(json);
    assert_eq!(format!("{json}"), r#""he\\\\\\\"llo""#);
}

#[test]
fn escape_json_string_escaped_escaped() {
    let json = SortedJson::serialize(r#"he\\"llo"#);
    let json = EscapedJson::from(json);
    assert_eq!(format!("{json}"), r#""he\\\\\\\\\\\"llo""#);
}

#[test]
fn number() {
    let json = SortedJson::serialize(3);
    let serialized = "3";
    check(json, serialized);
}

#[test]
fn boolean() {
    let json = SortedJson::serialize(true);
    let serialized = "true";
    check(json, serialized);
}

#[test]
fn string() {
    let json = SortedJson::serialize("he\"llo");
    let serialized = r#""he\"llo""#;
    check(json, serialized);
}

#[test]
fn serialize_array() {
    let json = SortedJson::serialize([3, 1, 2]);
    let serialized = "[3,1,2]";
    check(json, serialized);
}

#[test]
fn sorted_array() {
    let items = ["c", "a", "b"];
    let serialized = r#"["a","b","c"]"#;
    let items: Vec<SortedJson> = items.into_iter().map(SortedJson::serialize).collect();
    let json = SortedJson::array(items);
    check(json, serialized);
}

#[test]
fn nested_array() {
    let a = SortedJson::serialize(3);
    let b = SortedJson::serialize(2);
    let c = SortedJson::serialize(1);
    let d = SortedJson::serialize([1, 3, 2]);
    let json = SortedJson::array([a, b, c, d]);
    let serialized = r#"[1,2,3,[1,3,2]]"#;
    check(json, serialized);
}

#[test]
fn array_unsorted() {
    let items = ["c", "a", "b"];
    let serialized = r#"["c","a","b"]"#;
    let items: Vec<SortedJson> = items.into_iter().map(SortedJson::serialize).collect();
    let json = SortedJson::array_unsorted(items);
    check(json, serialized);
}
