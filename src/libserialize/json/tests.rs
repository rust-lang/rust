// Benchmarks and tests that require private items

extern crate test;
use test::Bencher;
use super::{from_str, Parser, StackElement, Stack};
use std::string;

#[test]
fn test_stack() {
    let mut stack = Stack::new();

    assert!(stack.is_empty());
    assert!(stack.is_empty());
    assert!(!stack.last_is_index());

    stack.push_index(0);
    stack.bump_index();

    assert!(stack.len() == 1);
    assert!(stack.is_equal_to(&[StackElement::Index(1)]));
    assert!(stack.starts_with(&[StackElement::Index(1)]));
    assert!(stack.ends_with(&[StackElement::Index(1)]));
    assert!(stack.last_is_index());
    assert!(stack.get(0) == StackElement::Index(1));

    stack.push_key("foo".to_string());

    assert!(stack.len() == 2);
    assert!(stack.is_equal_to(&[StackElement::Index(1), StackElement::Key("foo")]));
    assert!(stack.starts_with(&[StackElement::Index(1), StackElement::Key("foo")]));
    assert!(stack.starts_with(&[StackElement::Index(1)]));
    assert!(stack.ends_with(&[StackElement::Index(1), StackElement::Key("foo")]));
    assert!(stack.ends_with(&[StackElement::Key("foo")]));
    assert!(!stack.last_is_index());
    assert!(stack.get(0) == StackElement::Index(1));
    assert!(stack.get(1) == StackElement::Key("foo"));

    stack.push_key("bar".to_string());

    assert!(stack.len() == 3);
    assert!(stack.is_equal_to(&[StackElement::Index(1),
                                StackElement::Key("foo"),
                                StackElement::Key("bar")]));
    assert!(stack.starts_with(&[StackElement::Index(1)]));
    assert!(stack.starts_with(&[StackElement::Index(1), StackElement::Key("foo")]));
    assert!(stack.starts_with(&[StackElement::Index(1),
                                StackElement::Key("foo"),
                                StackElement::Key("bar")]));
    assert!(stack.ends_with(&[StackElement::Key("bar")]));
    assert!(stack.ends_with(&[StackElement::Key("foo"), StackElement::Key("bar")]));
    assert!(stack.ends_with(&[StackElement::Index(1),
                              StackElement::Key("foo"),
                              StackElement::Key("bar")]));
    assert!(!stack.last_is_index());
    assert!(stack.get(0) == StackElement::Index(1));
    assert!(stack.get(1) == StackElement::Key("foo"));
    assert!(stack.get(2) == StackElement::Key("bar"));

    stack.pop();

    assert!(stack.len() == 2);
    assert!(stack.is_equal_to(&[StackElement::Index(1), StackElement::Key("foo")]));
    assert!(stack.starts_with(&[StackElement::Index(1), StackElement::Key("foo")]));
    assert!(stack.starts_with(&[StackElement::Index(1)]));
    assert!(stack.ends_with(&[StackElement::Index(1), StackElement::Key("foo")]));
    assert!(stack.ends_with(&[StackElement::Key("foo")]));
    assert!(!stack.last_is_index());
    assert!(stack.get(0) == StackElement::Index(1));
    assert!(stack.get(1) == StackElement::Key("foo"));
}

#[bench]
fn bench_streaming_small(b: &mut Bencher) {
    b.iter( || {
        let mut parser = Parser::new(
            r#"{
                "a": 1.0,
                "b": [
                    true,
                    "foo\nbar",
                    { "c": {"d": null} }
                ]
            }"#.chars()
        );
        loop {
            match parser.next() {
                None => return,
                _ => {}
            }
        }
    });
}
#[bench]
fn bench_small(b: &mut Bencher) {
    b.iter( || {
        let _ = from_str(r#"{
            "a": 1.0,
            "b": [
                true,
                "foo\nbar",
                { "c": {"d": null} }
            ]
        }"#);
    });
}

fn big_json() -> string::String {
    let mut src = "[\n".to_string();
    for _ in 0..500 {
        src.push_str(r#"{ "a": true, "b": null, "c":3.1415, "d": "Hello world", "e": \
                        [1,2,3]},"#);
    }
    src.push_str("{}]");
    return src;
}

#[bench]
fn bench_streaming_large(b: &mut Bencher) {
    let src = big_json();
    b.iter( || {
        let mut parser = Parser::new(src.chars());
        loop {
            match parser.next() {
                None => return,
                _ => {}
            }
        }
    });
}
#[bench]
fn bench_large(b: &mut Bencher) {
    let src = big_json();
    b.iter( || { let _ = from_str(&src); });
}
