//@ check-pass
//@ compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(const_trait_impl, rustc_attrs)]

const trait Foo {
    #[rustc_do_not_const_check]
    fn into_iter(&self) {
        println!("FEAR ME!")
    }
}

const impl Foo for () {
    fn into_iter(&self) {
        // ^_^
    }
}

const _: () = Foo::into_iter(&());

fn main() {}
