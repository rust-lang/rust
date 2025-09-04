//@ only-aarch64
#[inline(always)]
//~^ ERROR cannot use `#[inline(always)]` with `#[target_feature]`
#[target_feature(enable="fp16")]
fn test() {

}

fn main() { }
