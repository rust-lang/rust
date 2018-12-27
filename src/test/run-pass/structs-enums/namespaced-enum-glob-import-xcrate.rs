// run-pass
// aux-build:namespaced_enums.rs

// pretty-expanded FIXME #23616

extern crate namespaced_enums;

fn _f(f: namespaced_enums::Foo) {
    use namespaced_enums::Foo::*;

    match f {
        A | B(_) | C { .. } => {}
    }
}

mod m {
    pub use namespaced_enums::Foo::*;
}

fn _f2(f: namespaced_enums::Foo) {
    match f {
        m::A | m::B(_) | m::C { .. } => {}
    }
}

pub fn main() {}
