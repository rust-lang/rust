// Test that impls for these two types are considered ovelapping:
//
// * `for<'r> fn(fn(&'r u32))`
// * `fn(fn(&'a u32)` where `'a` is free
//
// This is because, for `'a = 'static`, the two types overlap.
// Effectively for them to be equal to you get:
//
// * `for<'r> fn(fn(&'r u32)) <: fn(fn(&'static u32))`
//   * true if `exists<'r> { 'r: 'static }` (obviously true)
// * `fn(fn(&'static u32)) <: for<'r> fn(fn(&'r u32))`
//   * true if `forall<'r> { 'static: 'r }` (also true)

trait Trait {}

impl Trait for for<'r> fn(fn(&'r ())) {}
impl<'a> Trait for fn(fn(&'a ())) {}
//~^ ERROR conflicting implementations
//
// Note in particular that we do NOT get a future-compatibility warning
// here. This is because the new leak-check proposed in [MCP 295] does not
// "error" when these two types are equated.
//
// [MCP 295]: https://github.com/rust-lang/compiler-team/issues/295

fn main() {}
