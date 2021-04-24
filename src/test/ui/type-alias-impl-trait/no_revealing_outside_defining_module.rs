// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

fn main() {}

mod boo {
    pub type Boo = impl ::std::fmt::Debug;
    fn bomp() -> Boo {
        ""
    }
}

// We don't actually know the type here.

fn bomp2() {
    let _: &str = bomp(); //~ ERROR mismatched types
}

fn bomp() -> boo::Boo {
    "" //~ ERROR mismatched types
}

fn bomp_loop() -> boo::Boo {
    loop {}
}
