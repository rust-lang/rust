// aux-build:struct_variant_privacy.rs
extern crate struct_variant_privacy;

fn f(b: struct_variant_privacy::Bar) {
    //~^ ERROR enum `Bar` is private
    match b {
        struct_variant_privacy::Bar::Baz { a: _a } => {} //~ ERROR cannot match on
                                                         //~^ ERROR enum `Bar` is private
    }
}

fn main() {}
