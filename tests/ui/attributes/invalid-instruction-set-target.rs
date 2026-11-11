//@ only-x86_64-unknown-linux-gnu

#[instruction_set(arm::a32)]
//~^ ERROR target `x86_64-unknown-linux-gnu` does not support `#[instruction_set(arm::*)]`
fn main() {
}
