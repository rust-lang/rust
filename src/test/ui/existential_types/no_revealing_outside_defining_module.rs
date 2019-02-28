#![feature(existential_type)]

fn main() {}

mod boo {
    pub existential type Boo: ::std::fmt::Debug;
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
