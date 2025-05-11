#![feature(type_alias_impl_trait)]

fn main() {}

pub type Boo = impl ::std::fmt::Debug;
#[define_opaque(Boo)]
fn define() -> Boo {
    ""
}

// We don't actually know the type here.

fn bomp2() {
    let _: &str = bomp(); //~ ERROR mismatched types
}

fn bomp() -> Boo {
    "" //~ ERROR mismatched types
}

fn bomp_loop() -> Boo {
    loop {}
}
