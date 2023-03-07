const X: u8 = 1;
static Y: u8 = 1;
fn foo() {}

impl X {}
//~^ ERROR expected type, found constant `X`
impl Y {}
//~^ ERROR expected type, found static `Y`
impl foo {}
//~^ ERROR expected type, found function `foo`

fn main() {}
