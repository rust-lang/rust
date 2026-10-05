//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
//@[old] known-bug: #117392
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr

// This test previously caused an ICE in `fn crash` when relating opaque
// types. This has been fixed by the opaque type rework in the new solver.
pub trait BorrowComposite {
    type Ref<'a>: 'a;
}

impl BorrowComposite for () {
    type Ref<'a> = ();
}

pub trait Component<Args> {
    type Output;
}

impl<Args> Component<Args> for () {
    type Output = ();
}

pub fn delay<Args: BorrowComposite, Make: for<'a> FnMut(Args::Ref<'a>) -> C, C: Component<Args>>(
    make: Make,
) -> impl Component<Args> {
}

pub fn crash() -> impl Component<()> {
    delay(|()| delay(|()| ()))
}

pub fn main() {}
