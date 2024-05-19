fn foo1(s: &str) {
    s.as_str();
    //~^ ERROR no method named `as_str` found
}

fn foo2<'a>(s: &'a str) {
    s.as_str();
    //~^ ERROR no method named `as_str` found
}

fn foo3(s: &mut str) {
    s.as_str();
    //~^ ERROR no method named `as_str` found
}

fn foo4(s: &&str) {
    s.as_str();
    //~^ ERROR no method named `as_str` found
}

fn main() {}
