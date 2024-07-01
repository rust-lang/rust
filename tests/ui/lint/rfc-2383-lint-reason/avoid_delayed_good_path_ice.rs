//@ check-pass
//@only-16bit

#[expect(drop_bounds)]
fn trigger_rustc_lints<T: Drop>() {
}

fn main() {}
