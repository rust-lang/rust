#![feature(rustc_attrs)]

struct S;

impl S {
    #[rustc_confusables("bar")]
    fn foo() {}

    #[rustc_confusables("baz")]
    fn qux(&self, x: i32) {}
}

fn main() {
    S::bar();
    //~^ ERROR no associated function or constant named `bar`
    //~| HELP you might have meant to use `foo`

    let s = S;
    s.baz(10);
    //~^ ERROR no method named `baz`
    //~| HELP you might have meant to use `qux`
}
