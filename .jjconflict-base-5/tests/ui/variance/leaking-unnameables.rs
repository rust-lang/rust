// Test variance computation doesn't explode when we leak unnameable
// types due to `-> _` recovery.

pub struct Type<'a>(&'a ());

pub fn g() {}

pub fn f<T>() -> _ {
   //~^ ERROR the placeholder `_` is not allowed within types on item signatures
   g
}

fn main() {}
