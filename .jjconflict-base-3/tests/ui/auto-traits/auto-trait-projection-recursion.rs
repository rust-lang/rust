// Checking the `Send` bound in `main` requires:
//
// checking             <C<'static> as Y>::P: Send
// which normalizes to  Box<X<C<'?1>>>: Send
// which needs          X<C<'?1>>: Send
// which needs          <C<'?1> as Y>::P: Send
//
// At this point we used to normalize the predicate to `Box<X<C<'?2>>>: Send`
// and continue in a loop where we created new region variables to the
// recursion limit. To avoid this we now "canonicalize" region variables to
// lowest unified region vid. This means we instead have to prove
// `Box<X<C<'?1>>>: Send`, which we can because auto traits are coinductive.

//@ check-pass

// Avoid a really long error message if this regresses.
#![recursion_limit="20"]

trait Y {
    type P;
}

impl<'a> Y for C<'a> {
    type P = Box<X<C<'a>>>;
}

struct C<'a>(&'a ());
struct X<T: Y>(T::P);

fn is_send<S: Send>() {}

fn main() {
    is_send::<X<C<'static>>>();
}
