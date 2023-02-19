use super::*;

#[test]
fn empty() {
    assert_eq!(HtmlWithLimit::new(0).finish(), "");
    assert_eq!(HtmlWithLimit::new(60).finish(), "");
}

#[test]
fn basic() {
    let mut buf = HtmlWithLimit::new(60);
    buf.push("Hello ");
    buf.open_tag("em");
    buf.push("world");
    buf.close_tag();
    buf.push("!");
    assert_eq!(buf.finish(), "Hello <em>world</em>!");
}

#[test]
fn no_tags() {
    let mut buf = HtmlWithLimit::new(60);
    buf.push("Hello");
    buf.push(" world!");
    assert_eq!(buf.finish(), "Hello world!");
}

#[test]
fn limit_0() {
    let mut buf = HtmlWithLimit::new(0);
    buf.push("Hello ");
    buf.open_tag("em");
    buf.push("world");
    buf.close_tag();
    buf.push("!");
    assert_eq!(buf.finish(), "");
}

#[test]
fn exactly_limit() {
    let mut buf = HtmlWithLimit::new(12);
    buf.push("Hello ");
    buf.open_tag("em");
    buf.push("world");
    buf.close_tag();
    buf.push("!");
    assert_eq!(buf.finish(), "Hello <em>world</em>!");
}

#[test]
fn multiple_nested_tags() {
    let mut buf = HtmlWithLimit::new(60);
    buf.open_tag("p");
    buf.push("This is a ");
    buf.open_tag("em");
    buf.push("paragraph");
    buf.open_tag("strong");
    buf.push("!");
    buf.close_tag();
    buf.close_tag();
    buf.close_tag();
    assert_eq!(buf.finish(), "<p>This is a <em>paragraph<strong>!</strong></em></p>");
}

#[test]
fn forgot_to_close_tags() {
    let mut buf = HtmlWithLimit::new(60);
    buf.open_tag("p");
    buf.push("This is a ");
    buf.open_tag("em");
    buf.push("paragraph");
    buf.open_tag("strong");
    buf.push("!");
    assert_eq!(buf.finish(), "<p>This is a <em>paragraph<strong>!</strong></em></p>");
}

#[test]
fn past_the_limit() {
    let mut buf = HtmlWithLimit::new(20);
    buf.open_tag("p");
    (0..10).try_for_each(|n| {
        buf.open_tag("strong");
        buf.push("word#")?;
        buf.push(&n.to_string())?;
        buf.close_tag();
        ControlFlow::Continue(())
    });
    buf.close_tag();
    assert_eq!(
        buf.finish(),
        "<p>\
             <strong>word#0</strong>\
             <strong>word#1</strong>\
             <strong>word#2</strong>\
             </p>"
    );
}

#[test]
fn quickly_past_the_limit() {
    let mut buf = HtmlWithLimit::new(6);
    buf.open_tag("p");
    buf.push("Hello");
    buf.push(" World");
    // intentionally not closing <p> before finishing
    assert_eq!(buf.finish(), "<p>Hello</p>");
}

#[test]
fn close_too_many() {
    let mut buf = HtmlWithLimit::new(60);
    buf.open_tag("p");
    buf.push("Hello");
    buf.close_tag();
    // This call does not panic because there are valid cases
    // where `close_tag()` is called with no tags left to close.
    // So `close_tag()` does nothing in this case.
    buf.close_tag();
    assert_eq!(buf.finish(), "<p>Hello</p>");
}
