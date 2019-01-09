#[start]
fn foo(_: isize, _: *const *const u8) -> isize { 0 }
//~^ ERROR a #[start] function is an experimental feature
