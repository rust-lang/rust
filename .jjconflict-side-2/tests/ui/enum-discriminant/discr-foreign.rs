//@ aux-build:discr-foreign-dep.rs
//@ build-pass

extern crate discr_foreign_dep;

fn main() {
    match Default::default() {
        discr_foreign_dep::Foo::A(_) => {}
        _ => {}
    }
}
