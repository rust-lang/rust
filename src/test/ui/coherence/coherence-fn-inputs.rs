// Test that we consider these two types completely equal:
//
// * `for<'a, 'b> fn(&'a u32, &'b u32)`
// * `for<'c> fn(&'c u32, &'c u32)`
//
// For a long time we considered these to be distinct types. But in fact they
// are equivalent, if you work through the implications of subtyping -- this is
// because:
//
// * `'c` can be the intersection of `'a` and `'b` (and there is always an intersection)
// * `'a` and `'b` can both be equal to `'c`
//
// check-pass

trait Trait {}
impl Trait for for<'a, 'b> fn(&'a u32, &'b u32) {}
impl Trait for for<'c> fn(&'c u32, &'c u32) {}

fn main() {}
