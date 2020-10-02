// check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))] //[full]~WARN the feature `const_generics` is incomplete
#![cfg_attr(min, feature(min_const_generics))]

struct Generic<const V: usize>;

fn main() {
    let _ = Generic::<0>;
}
