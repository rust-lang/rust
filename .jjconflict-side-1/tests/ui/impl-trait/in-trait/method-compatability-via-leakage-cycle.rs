//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ known-bug: #139788

// Recursively using the trait method inside of an impl in case checking
// method compatability relies on opaque type leakage currently causes a
// cycle error.

trait Trait {
    // desugars to
    // type Assoc: Sized + Send;
    // fn foo(b: bool) -> Self::Assoc;
    fn foo(b: bool) -> impl Sized + Send;
}

impl Trait for u32 {
    // desugars to
    // type Assoc = impl_rpit::<Self>;
    // fn foo(b: bool) -> Self::Assoc { .. }
    fn foo(b: bool) -> impl Sized {
        if b {
            u32::foo(false)
        } else {
            1u32
        }
    }
}

fn main() {}
