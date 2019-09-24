fn foo1(s: &str) {
    s.as_str();
    //~^ ERROR no method named `as_str` found for type `&str` in the current scope
}

fn foo2<'a>(s: &'a str) {
    s.as_str();
    //~^ ERROR no method named `as_str` found for type `&'a str` in the current scope
}

fn foo3(s: &mut str) {
    s.as_str();
    //~^ ERROR no method named `as_str` found for type `&mut str` in the current scope
}

fn foo4(s: &&str) {
    s.as_str();
    //~^ ERROR no method named `as_str` found for type `&&str` in the current scope
}

fn main() {}
