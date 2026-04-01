use std::str::FromStr;

use super::super::sorted_template::*;

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
    let mut template = SortedTemplate::<Html>::from_before_after("", "");
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
    let mut template = SortedTemplate::<Html>::from_before_after(before, after);
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
    let mut template = SortedTemplate::<Js>::from_before_after("", "");
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
    let template = SortedTemplate::<Js>::from_before_after("[", "]");
    let template = format!("{template}");
    let (template, end) = template.rsplit_once("\n").unwrap();
    assert_eq!(template, format!("[]"));
    assert!(is_comment_js(end));
    assert!(!end.contains("\n"));
}

#[test]
fn js_number_array() {
    let inserts = ["1", "2", "3"];
    let mut template = SortedTemplate::<Js>::from_before_after("[", "]");
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
    let mut template = SortedTemplate::<Js>::from_template("[#]", "#").unwrap();
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
    let mut template = SortedTemplate::<Js>::from_before_after("[", "]");
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
    let mut template = SortedTemplate::<Html>::from_before_after(before, after);
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
    let template = SortedTemplate::<Js>::from_before_after("", "");
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
