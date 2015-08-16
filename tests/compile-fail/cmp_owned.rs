#![feature(plugin)]
#![plugin(clippy)]

#[deny(cmp_owned)]
fn main() {
    let x = "oh";

    #[allow(str_to_string)]
    fn with_to_string(x : &str) {
        x != "foo".to_string(); //~ERROR this creates an owned instance
    }
    with_to_string(x);

    x != "foo".to_owned(); //~ERROR this creates an owned instance

    // removed String::from_str(..), as it has finally been removed in 1.4.0
    // as of 2015-08-14

    x != String::from("foo"); //~ERROR this creates an owned instance
}
