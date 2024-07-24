use std::cmp::Ordering;

use super::print_item::compare_names;
use super::{AllTypes, Buffer};

#[test]
fn test_compare_names() {
    for &(a, b) in &[
        ("hello", "world"),
        ("", "world"),
        ("123", "hello"),
        ("123", ""),
        ("123test", "123"),
        ("hello", ""),
        ("hello", "hello"),
        ("hello123", "hello123"),
        ("hello123", "hello12"),
        ("hello12", "hello123"),
        ("hello01abc", "hello01xyz"),
        ("hello0abc", "hello0"),
        ("hello0", "hello0abc"),
        ("01", "1"),
    ] {
        assert_eq!(compare_names(a, b), a.cmp(b), "{:?} - {:?}", a, b);
    }
    assert_eq!(compare_names("u8", "u16"), Ordering::Less);
    assert_eq!(compare_names("u32", "u16"), Ordering::Greater);
    assert_eq!(compare_names("u8_to_f64", "u16_to_f64"), Ordering::Less);
    assert_eq!(compare_names("u32_to_f64", "u16_to_f64"), Ordering::Greater);
    assert_eq!(compare_names("u16_to_f64", "u16_to_f64"), Ordering::Equal);
    assert_eq!(compare_names("u16_to_f32", "u16_to_f64"), Ordering::Less);
}

#[test]
fn test_name_sorting() {
    let names = [
        "Apple", "Banana", "Fruit", "Fruit0", "Fruit00", "Fruit01", "Fruit02", "Fruit1", "Fruit2",
        "Fruit20", "Fruit30x", "Fruit100", "Pear",
    ];
    let mut sorted = names.to_owned();
    sorted.sort_by(|&l, r| compare_names(l, r));
    assert_eq!(names, sorted);
}

#[test]
fn test_all_types_prints_header_once() {
    // Regression test for #82477
    let all_types = AllTypes::new();

    let mut buffer = Buffer::new();
    all_types.print(&mut buffer);

    assert_eq!(1, buffer.into_inner().matches("List of all items").count());
}

mod sorted_json {
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
}

mod sorted_template {
    use super::super::sorted_template::*;
    use std::str::FromStr;

    fn is_comment_js(s: &str) -> bool {
        s.starts_with("//")
    }

    fn is_comment_html(s: &str) -> bool {
        // not correct but good enough for these tests
        s.starts_with("<!--") && s.ends_with("-->")
    }

    #[test]
    fn html_from_empty() {
        let inserts = ["<p>hello</p>", "<p>kind</p>", "<p>hello</p>", "<p>world</p>"];
        let mut template = SortedTemplate::<Html>::before_after("", "");
        for insert in inserts {
            template.append(insert.to_string());
        }
        let template = format!("{template}");
        let (template, end) = template.rsplit_once("\n").unwrap();
        assert_eq!(template, "<p>hello</p><p>kind</p><p>world</p>");
        assert!(is_comment_html(end));
        assert!(!end.contains("\n"));
    }

    #[test]
    fn html_page() {
        let inserts = ["<p>hello</p>", "<p>kind</p>", "<p>world</p>"];
        let before = "<html><head></head><body>";
        let after = "</body>";
        let mut template = SortedTemplate::<Html>::before_after(before, after);
        for insert in inserts {
            template.append(insert.to_string());
        }
        let template = format!("{template}");
        let (template, end) = template.rsplit_once("\n").unwrap();
        assert_eq!(template, format!("{before}{}{after}", inserts.join("")));
        assert!(is_comment_html(end));
        assert!(!end.contains("\n"));
    }

    #[test]
    fn js_from_empty() {
        let inserts = ["1", "2", "2", "2", "3", "1"];
        let mut template = SortedTemplate::<Js>::before_after("", "");
        for insert in inserts {
            template.append(insert.to_string());
        }
        let template = format!("{template}");
        let (template, end) = template.rsplit_once("\n").unwrap();
        assert_eq!(template, "1,2,3");
        assert!(is_comment_js(end));
        assert!(!end.contains("\n"));
    }

    #[test]
    fn js_empty_array() {
        let template = SortedTemplate::<Js>::before_after("[", "]");
        let template = format!("{template}");
        let (template, end) = template.rsplit_once("\n").unwrap();
        assert_eq!(template, format!("[]"));
        assert!(is_comment_js(end));
        assert!(!end.contains("\n"));
    }

    #[test]
    fn js_number_array() {
        let inserts = ["1", "2", "3"];
        let mut template = SortedTemplate::<Js>::before_after("[", "]");
        for insert in inserts {
            template.append(insert.to_string());
        }
        let template = format!("{template}");
        let (template, end) = template.rsplit_once("\n").unwrap();
        assert_eq!(template, format!("[1,2,3]"));
        assert!(is_comment_js(end));
        assert!(!end.contains("\n"));
    }

    #[test]
    fn magic_js_number_array() {
        let inserts = ["1", "1"];
        let mut template = SortedTemplate::<Js>::magic("[#]", "#").unwrap();
        for insert in inserts {
            template.append(insert.to_string());
        }
        let template = format!("{template}");
        let (template, end) = template.rsplit_once("\n").unwrap();
        assert_eq!(template, format!("[1]"));
        assert!(is_comment_js(end));
        assert!(!end.contains("\n"));
    }

    #[test]
    fn round_trip_js() {
        let inserts = ["1", "2", "3"];
        let mut template = SortedTemplate::<Js>::before_after("[", "]");
        for insert in inserts {
            template.append(insert.to_string());
        }
        let template1 = format!("{template}");
        let mut template = SortedTemplate::<Js>::from_str(&template1).unwrap();
        assert_eq!(template1, format!("{template}"));
        template.append("4".to_string());
        let template = format!("{template}");
        let (template, end) = template.rsplit_once("\n").unwrap();
        assert_eq!(template, "[1,2,3,4]");
        assert!(is_comment_js(end));
    }

    #[test]
    fn round_trip_html() {
        let inserts = ["<p>hello</p>", "<p>kind</p>", "<p>world</p>", "<p>kind</p>"];
        let before = "<html><head></head><body>";
        let after = "</body>";
        let mut template = SortedTemplate::<Html>::before_after(before, after);
        template.append(inserts[0].to_string());
        template.append(inserts[1].to_string());
        let template = format!("{template}");
        let mut template = SortedTemplate::<Html>::from_str(&template).unwrap();
        template.append(inserts[2].to_string());
        let template = format!("{template}");
        let (template, end) = template.rsplit_once("\n").unwrap();
        assert_eq!(template, format!("{before}<p>hello</p><p>kind</p><p>world</p>{after}"));
        assert!(is_comment_html(end));
    }

    #[test]
    fn blank_js() {
        let inserts = ["1", "2", "3"];
        let template = SortedTemplate::<Js>::before_after("", "");
        let template = format!("{template}");
        let (t, _) = template.rsplit_once("\n").unwrap();
        assert_eq!(t, "");
        let mut template = SortedTemplate::<Js>::from_str(&template).unwrap();
        for insert in inserts {
            template.append(insert.to_string());
        }
        let template1 = format!("{template}");
        let mut template = SortedTemplate::<Js>::from_str(&template1).unwrap();
        assert_eq!(template1, format!("{template}"));
        template.append("4".to_string());
        let template = format!("{template}");
        let (template, end) = template.rsplit_once("\n").unwrap();
        assert_eq!(template, "1,2,3,4");
        assert!(is_comment_js(end));
    }
}
