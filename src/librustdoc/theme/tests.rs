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
    get_differences(&load_css_paths(against.as_bytes()),
                    &load_css_paths(text.as_bytes()),
                    &mut ret);
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
    let paths = load_css_paths(text.as_bytes());
    assert!(paths.children.contains(&CssPath::new("a b c d".to_owned())));
}

#[test]
fn test_comparison() {
    let x = r#"
a {
    b {
        c {}
    }
}
"#;

    let y = r#"
a {
    b {}
}
"#;

    let against = load_css_paths(y.as_bytes());
    let other = load_css_paths(x.as_bytes());

    let mut ret = Vec::new();
    get_differences(&against, &other, &mut ret);
    assert!(ret.is_empty());
    get_differences(&other, &against, &mut ret);
    assert_eq!(ret, vec!["  Missing \"c\" rule".to_owned()]);
}

#[test]
fn check_empty_css() {
    let events = load_css_events(&[]);
    assert_eq!(events.len(), 0);
}

#[test]
fn check_invalid_css() {
    let events = load_css_events(b"*");
    assert_eq!(events.len(), 0);
}
