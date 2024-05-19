#![feature(impl_trait_in_assoc_type)]

//@ revisions: ok bad
//@ [ok] check-pass

trait Foo {
    type Assoc;
    fn bar() -> Self::Assoc;
}

impl Foo for () {
    type Assoc = impl Sized;
    fn bar() -> Self::Assoc {
        let closure = || -> Self::Assoc {
            #[cfg(ok)]
            let x: Self::Assoc = 42_i32;
            #[cfg(bad)]
            let x: Self::Assoc = ();
            x
        };
        let _: i32 = closure(); //[bad]~ ERROR mismatched types
        1i32 //[bad]~ ERROR mismatched types
    }
}

fn main() {}
