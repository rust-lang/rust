type IMPL_T<A> = option<A>;

pure fn EACH<A>(self: IMPL_T<A>, f: fn(A) -> bool) {
    match self {
      none => (),
      some(a) => { f(a); }
    }
}

fn SIZE_HINT<A>(self: IMPL_T<A>) -> option<uint> {
    match self {
      none => some(0u),
      some(_) => some(1u)
    }
}
