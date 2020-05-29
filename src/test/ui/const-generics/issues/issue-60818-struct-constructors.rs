// check-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

struct Generic<const V: usize>;

fn main() {
    let _ = Generic::<0>;
}
