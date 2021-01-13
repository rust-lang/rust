// check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))] //[full]~WARN the feature `const_generics` is incomplete

struct Generic<const V: usize>;

fn main() {
    let _ = Generic::<0>;
}
