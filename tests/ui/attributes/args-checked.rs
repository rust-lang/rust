#[inline(always = 5)]
//~^ ERROR malformed
#[inline(always(x, y, z))]
//~^ ERROR malformed
fn main() {

}