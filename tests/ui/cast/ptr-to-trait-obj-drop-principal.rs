//! Test that non-coercion casts aren't allowed to drop the principal,
//! because they cannot modify the pointer metadata.
//!
//! We test this in a const context to guard against UB if this is allowed
//! in the future.

trait Trait {}
impl Trait for () {}

struct Wrapper<T: ?Sized>(T);

const OBJECT: *const (dyn Trait + Send) = &();

// coercions are allowed
const _: *const dyn Send = OBJECT as _;

// casts are **not** allowed
const _: *const Wrapper<dyn Send> = OBJECT as _;
//~^ ERROR casting `*const (dyn Trait + Send + 'static)` as `*const Wrapper<dyn Send>` is invalid

fn main() {}
