struct HasDyn {
    field: &'static dyn std::fmt::Debug,
}

struct Hide(HasDyn);

const fn no_inner_dyn_trait(_x: Hide) {}
const fn no_inner_dyn_trait2(x: Hide) {
    x.0.field;
//~^ ERROR trait bounds other than `Sized`
}
const fn no_inner_dyn_trait_ret() -> Hide { Hide(HasDyn { field: &0 }) }
//~^ ERROR trait bounds other than `Sized`
//~| WARNING temporary value dropped while borrowed
//~| WARNING this error has been downgraded to a warning
//~| WARNING this warning will become a hard error in the future

fn main() {}
