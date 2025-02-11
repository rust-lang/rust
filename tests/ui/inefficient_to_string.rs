#![deny(clippy::inefficient_to_string)]

use std::borrow::Cow;

fn main() {
    let rstr: &str = "hello";
    let rrstr: &&str = &rstr;
    let rrrstr: &&&str = &rrstr;
    let _: String = rstr.to_string();
    let _: String = rrstr.to_string();
    //~^ inefficient_to_string
    let _: String = rrrstr.to_string();
    //~^ inefficient_to_string

    let string: String = String::from("hello");
    let rstring: &String = &string;
    let rrstring: &&String = &rstring;
    let rrrstring: &&&String = &rrstring;
    let _: String = string.to_string();
    let _: String = rstring.to_string();
    let _: String = rrstring.to_string();
    //~^ inefficient_to_string
    let _: String = rrrstring.to_string();
    //~^ inefficient_to_string

    let cow: Cow<'_, str> = Cow::Borrowed("hello");
    let rcow: &Cow<'_, str> = &cow;
    let rrcow: &&Cow<'_, str> = &rcow;
    let rrrcow: &&&Cow<'_, str> = &rrcow;
    let _: String = cow.to_string();
    let _: String = rcow.to_string();
    let _: String = rrcow.to_string();
    //~^ inefficient_to_string
    let _: String = rrrcow.to_string();
    //~^ inefficient_to_string
}
