//@ check-pass
// When we changed from lazy resolving to eager resolving, this was in some cases breaking
// We introduced a workaround in https://github.com/rust-lang/rust/pull/158447 to more often
// avoid breakage. This workaround was not perfect, and we knew this when landing,
// we just had no examples that showed the breakage until this one.
//
// The workaround would try to fudge a formally expected type, but if no structural changes
// to the type happened, would rollback the fudging again, and stuck to the formal expected type.
//
// This rollback behavior worked per function argument, and that's what goes wrong in this
// reproducer: g() is called with two parameters. The 2nd parameter gets fudged, and has a
// structural change (the return type is inferred to unit), so fudging gets accepted,
// not rolled back. In the process, the input of the closure, `x` becomes an unconstrained
// type variable, and type inference fails, despite us knowing that its type is `X` already,
// by the difinion of `g`.

fn f<T>() -> X {
    let v = h(g(X, |x| x.m()));
    //         ^^^^^^^^^^^^^^ two parameters
    v
}

fn g<T, U>(_: T, _: fn(T) -> U) -> Y<T, U> {
    loop {}
}

fn h<U>(_: Y<U, ()>) -> U {
    loop {}
}

struct X;

impl X {
    fn m(self) {}
}

struct Y<T, U> {
    a: fn(T) -> T,
    b: U,
}

fn main() {}
