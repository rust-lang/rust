#[inline(always = 5)]
//~^ ERROR malformed
#[inline(always(x, y, z))]
//~^ ERROR malformed
#[instruction_set(arm::a32 = 5)]
//~^ ERROR malformed
#[instruction_set(arm::a32(x, y, z))]
//~^ ERROR malformed
fn main() {

}