//@ check-pass

struct HasDyn {
    field: &'static dyn std::fmt::Debug,
}

struct Hide(HasDyn);

const fn no_inner_dyn_trait(_x: Hide) {}
const fn no_inner_dyn_trait2(x: Hide) {
    x.0.field;
}
const fn no_inner_dyn_trait_ret() -> Hide { Hide(HasDyn { field: &0 }) }

fn main() {}
