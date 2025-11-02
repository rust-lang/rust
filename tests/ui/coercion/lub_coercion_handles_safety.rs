//@ check-pass

//@ only-x86_64
//  because target features

macro_rules! lub {
    ($lhs:expr, $rhs:expr) => {
        if true { $lhs } else { $rhs }
    };
}

fn safety_lub() {
    unsafe fn lhs() {}
    fn rhs() {}

    // We have two different fn defs, the only valid lub here
    // is to go to fnptrs. However, in order to go to fnptrs
    // `rhs` must coerce from a *safe* function to an *unsafe*
    // one.
    let lubbed = lub!(lhs, rhs);
    let lubbed: unsafe fn() = lubbed;
}

#[target_feature(enable = "sse2")]
fn target_feature_aware_safety_lub() {
    #[target_feature(enable = "sse2")]
    fn lhs() {}
    fn rhs() {}
    unsafe fn rhs_unsafe() {}

    // We have two different fn defs, the only valid lub here
    // is to go to fnptrs. However, in order to go to fnptrs
    // `lhs` must coerce from an unsafe fn to a safe one due
    // to the correct target features being enabled
    let lubbed = lub!(lhs, rhs);
    let lubbed: fn() = lubbed;

    // Similar case here except we must recognise that rhs
    // is an unsafe fn so lhs must be an unsafe fn even though
    // it *could* be safe
    let lubbed = lub!(lhs, rhs_unsafe);
    let lubbed: unsafe fn() = lubbed;
}

fn main() {}
