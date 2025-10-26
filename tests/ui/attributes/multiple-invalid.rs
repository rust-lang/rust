// This test checks that all expected errors occur when there are multiple invalid attributes
// on an item.

#[inline]
//~^ ERROR attribute cannot be used on
#[target_feature(enable = "sse2")]
//~^ ERROR attribute cannot be used on
const FOO: u8 = 0;

fn main() { }
