// Test for #151358, assertion failed: !worker_thread.is_null()
//~^ ERROR cycle detected when getting the resolver for lowering

trait Default {}
use std::num::NonZero;
fn main() {
    NonZero();
    todo!();
}
