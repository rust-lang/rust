// In this regression test we check that a path pattern referring to a unit variant
// through a type alias is successful in inferring the generic argument.

// build-pass (FIXME(62277): could be check-pass?)

enum Opt<T> {
    N,
    S(T),
}

type OptAlias<T> = Opt<T>;

fn f1(x: OptAlias<u8>) {
    match x {
        OptAlias::N // We previously failed to infer `T` to `u8`.
            => (),
        _ => (),
    }

    match x {
        <
            OptAlias<_> // And we failed to infer this type also.
        >::N => (),
        _ => (),
    }
}

fn main() {}
