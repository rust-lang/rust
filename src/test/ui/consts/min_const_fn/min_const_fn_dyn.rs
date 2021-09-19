struct HasDyn {
    field: &'static dyn std::fmt::Debug,
}

struct Hide(HasDyn);

const fn no_inner_dyn_trait(_x: Hide) {}
const fn no_inner_dyn_trait2(x: Hide) {
    x.0.field;
//~^ ERROR trait objects in const fn are unstable
}
const fn no_inner_dyn_trait_ret() -> Hide { Hide(HasDyn { field: &0 }) }
//~^ ERROR trait objects in const fn are unstable

fn main() {}
