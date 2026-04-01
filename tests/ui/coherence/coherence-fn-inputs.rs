//@ check-pass

// These types were previously considered equal as they are subtypes of each other.
// This has been changed in #118247 and we now consider them to be disjoint.
//
// * `for<'a, 'b> fn(&'a u32, &'b u32)`
// * `for<'c> fn(&'c u32, &'c u32)`
//
// These types are subtypes of each other as:
//
// * `'c` can be the intersection of `'a` and `'b` (and there is always an intersection)
// * `'a` and `'b` can both be equal to `'c`

trait Trait {}
impl Trait for for<'a, 'b> fn(&'a u32, &'b u32) {}
impl Trait for for<'c> fn(&'c u32, &'c u32) {
    //~^ WARN conflicting implementations of trait `Trait` for type `for<'a, 'b> fn(&'a u32, &'b u32)` [coherence_leak_check]
    //~| WARN the behavior may change in a future release
    //
    // Note in particular that we do NOT get a future-compatibility warning
    // here. This is because the new leak-check proposed in [MCP 295] does not
    // "error" when these two types are equated.
    //
    // [MCP 295]: https://github.com/rust-lang/compiler-team/issues/295
}

fn main() {}
