//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

// A test showcasing a behavior change in cycle detection between
// the old and new solver. The old solver detects cycles in both
// fulfill and in evaluate. The cycle detection in fulfillment
// does not resolve regions. Each occurance of the `Foo<'a>: Send`
// impl creates a fresh inference variable even if it is later
// constrainted to `'static`. This means we never detect a cycle.
//
// This is not an issue with builtin auto-trait impls as they don't
// create impl args instead, simply using the generic arguments of
// the self type directly.
//
// The new solver properly resolves regions. The old solver does resolve
// regions in `project`, so there we do detect cycles even if there are
// fresh region variables.

struct Foo<'a>(&'a ());

unsafe impl<'a> Send for Foo<'a>
where
    Foo<'a>: Send,
{}

fn require<T: Send>() {}

fn main() {
    require::<Foo<'static>>();
    //[current]~^ ERROR: overflow evaluating the requirement `Foo<'_>: Send`
}
