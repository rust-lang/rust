//@ run-pass
//@ aux-build:namespaced_enums.rs


extern crate namespaced_enums;

use namespaced_enums::Foo;

fn _foo (f: Foo) {
    match f {
        Foo::A | Foo::B(_) | Foo::C { .. } => {}
    }
}

pub fn main() {}
