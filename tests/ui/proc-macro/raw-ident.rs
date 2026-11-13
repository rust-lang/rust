//@ proc-macro: raw-ident.rs

#[macro_use] extern crate raw_ident;

make_raw_module!(r#crate);
make_raw_module!(r#self);
make_raw_module!(r#Self);
make_raw_module!(r#super);

fn main() {
    make_struct!(fn);
    make_struct!(Foo);
    make_struct!(await);

    r#fn;
    r#Foo;
    Foo;
    r#await;

    make_bad_struct!(S); //~ ERROR expected one of

    let _: Option<r#crate::Marker> = None;
    let _: Option<r#self::Marker> = None;
    let _: Option<r#Self::Marker> = None;
    let _: Option<r#super::Marker> = None;
}
