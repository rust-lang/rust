struct S {}

impl S {
    fn foo(&mur Self) {}
    //~^ ERROR expected one of `:`, `@`
    fn bar(&'static mur Self) {}
    //~^ ERROR unexpected lifetime
    //~| ERROR expected one of `:`, `@`

    fn baz(&mur Self @ _) {}
    //~^ ERROR expected one of `:`, `@`
}

fn main() {}
