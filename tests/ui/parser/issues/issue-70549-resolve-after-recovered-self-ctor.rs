struct S {}

impl S {
    fn foo(&mur Self) {}
    //~^ ERROR expected identifier, found keyword `Self`
    //~| ERROR expected one of `:`, `@`
    //~| ERROR expected unit struct, found self constructor `Self`
    fn bar(&'static mur Self) {}
    //~^ ERROR unexpected lifetime
    //~| ERROR expected identifier, found keyword `Self`
    //~| ERROR expected one of `:`, `@`
    //~| ERROR expected unit struct, found self constructor `Self`

    fn baz(&mur Self @ _) {}
    //~^ ERROR expected one of `:`, `@`
}

fn main() {}
