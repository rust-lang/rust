//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// A projection error is only redundant when the trait goal it rests on failed too.
// Here `S: Super` holds, so `<S as Super>::Assoc == u32` failed on its own and has to
// be reported even though the `S: Sub` error at the same span elaborates to `S: Super`.

trait Super {
    type Assoc;
}
trait Sub: Super {}

struct S;

impl Super for S {
    type Assoc = u8;
}

fn f<T>(_: T)
where
    T: Sub,
    T: Super<Assoc = u32>,
{
}

fn main() {
    f(S);
    //~^ ERROR the trait bound `S: Sub` is not satisfied
    //~| ERROR type mismatch resolving `<S as Super>::Assoc == u32`
}
