// build-pass (FIXME(62277): could be check-pass?)

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

struct Generic<const V: usize>;

fn main() {
    let _ = Generic::<0>;
}
