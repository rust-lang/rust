//@ proc-macro: raw-ident.rs

#[macro_use] extern crate raw_ident;

fn main() {
    make_struct!(fn);
    make_struct!(Foo);
    make_struct!(await);

    r#fn;
    r#Foo;
    Foo;
    r#await;

    make_bad_struct!(S); //~ ERROR expected one of
}
