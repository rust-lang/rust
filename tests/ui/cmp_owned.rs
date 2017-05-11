#![feature(plugin)]
#![plugin(clippy)]

#[deny(cmp_owned)]
#[allow(unnecessary_operation)]
fn main() {
    fn with_to_string(x : &str) {
        x != "foo".to_string();

        "foo".to_string() != x;
    }

    let x = "oh";

    with_to_string(x);

    x != "foo".to_owned();

    // removed String::from_str(..), as it has finally been removed in 1.4.0
    // as of 2015-08-14

    x != String::from("foo");

    42.to_string() == "42";
}
