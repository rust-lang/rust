#[rustc_safe_intrinsic]
//~^ ERROR the `#[rustc_safe_intrinsic]` attribute is used internally to mark intrinsics as safe
//~| ERROR attribute should be applied to intrinsic functions
fn safe() {}

fn main() {}
