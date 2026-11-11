//@ revisions: current next
//@[current] check-pass
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// The only use of the opaque below is the recursive call, so nothing ever
// constrains its hidden type. The error has to say that the ambiguity is in
// the opaque's hidden type, otherwise it reads as an ordinary inference
// failure and gives no hint that an opaque is involved at all.
//
// Only the next solver reports this; the old solver accepts the item.

#![allow(unconditional_recursion)]

fn recursive_rpit() -> impl Sized {
    //[next]~^ ERROR type annotations needed
    recursive_rpit()
}

fn main() {}
