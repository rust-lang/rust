// Test for #151358, assertion failed: !worker_thread.is_null()
//~^ ERROR internal compiler error: query cycle when printing cycle detected
//~^^ ERROR cycle detected when getting the resolver for lowering

trait Default {}
use std::num::NonZero;
fn main() {
    NonZero();
    todo!();
}
