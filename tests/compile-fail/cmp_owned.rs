#![feature(plugin, collections)]
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

    #[allow(deprecated)] // for from_str
    fn old_timey(x : &str) {
        x != String::from_str("foo"); //~ERROR this creates an owned instance
    }
    old_timey(x);

    x != String::from("foo"); //~ERROR this creates an owned instance
}
