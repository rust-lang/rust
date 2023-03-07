#[start]
fn foo(_: isize, _: *const *const u8) -> isize { 0 }
//~^ ERROR `#[start]` functions are experimental
