use super::*;

#[test]
fn test_comments_in_rules() {
    let text = r#"
rule a {}

rule b, c
// a line comment
{}

rule d
// another line comment
e {}

rule f/* a multine

comment*/{}

rule g/* another multine

comment*/h

i {}

rule j/*commeeeeent

you like things like "{}" in there? :)
*/
end {}"#;

    let against = r#"
rule a {}

rule b, c {}

rule d e {}

rule f {}

rule gh i {}

rule j end {}
"#;

    let mut ret = Vec::new();
    get_differences(&load_css_paths(against).unwrap(), &load_css_paths(text).unwrap(), &mut ret);
    assert!(ret.is_empty());
}

#[test]
fn test_text() {
    let text = r#"
a
/* sdfs
*/ b
c // sdf
d {}
"#;
    let paths = load_css_paths(text).unwrap();
    assert!(paths.contains_key(&"a b c d".to_owned()));
}

#[test]
fn test_comparison() {
    let x = r#"
@a {
    b {}
}
"#;

    let y = r#"
@a {
    b {}
    c {}
}
"#;

    let against = load_css_paths(y).unwrap();
    let other = load_css_paths(x).unwrap();

    let mut ret = Vec::new();
    get_differences(&against, &other, &mut ret);
    assert!(ret.is_empty());
    get_differences(&other, &against, &mut ret);
    assert_eq!(ret, vec!["  Missing rule `c`".to_owned()]);
}

#[test]
fn check_empty_css() {
    let paths = load_css_paths("").unwrap();
    assert_eq!(paths.len(), 0);
}

#[test]
fn check_invalid_css() {
    let paths = load_css_paths("*").unwrap();
    assert_eq!(paths.len(), 0);
}

#[test]
fn test_with_minification() {
    let text = include_str!("../html/static/css/themes/dark.css");
    let minified = minifier::css::minify(&text).expect("CSS minification failed").to_string();

    let against = load_css_paths(text).unwrap();
    let other = load_css_paths(&minified).unwrap();

    let mut ret = Vec::new();
    get_differences(&against, &other, &mut ret);
    assert!(ret.is_empty());
}

#[test]
fn test_media() {
    let text = r#"
@media (min-width: 701px) {
    a:hover {
        color: #fff;
    }

    b {
        x: y;
    }
}
"#;

    let paths = load_css_paths(text).unwrap();
    eprintln!("{:?}", paths);
    let p = paths.get("@media (min-width:701px)");
    assert!(p.is_some());
    let p = p.unwrap();
    assert!(p.children.get("a:hover").is_some());
    assert!(p.children.get("b").is_some());
}
