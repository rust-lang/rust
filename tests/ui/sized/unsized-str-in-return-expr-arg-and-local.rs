fn foo() -> impl Sized {
//~^ ERROR the size for values of type `str` cannot be known at compilation time
//~| HELP the trait `Sized` is not implemented for `str`
    *"" //~ HELP consider not dereferencing the expression
}
fn bar(_: impl Sized) {}
struct S;

impl S {
    fn baz(&self, _: impl Sized) {}
}

fn main() {
    let _ = foo();
    let x = *"";
    //~^ ERROR the size for values of type `str` cannot be known at compilation time
    //~| HELP consider not dereferencing the expression
    //~| HELP the trait `Sized` is not implemented for `str`
    bar(x);
    S.baz(x);
    bar(*"");
    //~^ ERROR the size for values of type `str` cannot be known at compilation time
    //~| HELP consider not dereferencing the expression
    //~| HELP the trait `Sized` is not implemented for `str`
    S.baz(*"");
    //~^ ERROR the size for values of type `str` cannot be known at compilation time
    //~| HELP consider not dereferencing the expression
    //~| HELP the trait `Sized` is not implemented for `str`
}
