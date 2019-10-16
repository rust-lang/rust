#![feature(matches_macro)]

fn main() {
    let foo = 'f';
    assert!(matches!(foo, 'A'..='Z' | 'a'..='z'));
    //~^ Error: cannot find macro `matches` in this scope
}
