type IMPL_T<A> = Option<A>;

pure fn EACH<A>(self: IMPL_T<A>, f: fn(A) -> bool) {
    match self {
      None => (),
      Some(a) => { f(a); }
    }
}

pure fn SIZE_HINT<A>(self: IMPL_T<A>) -> Option<uint> {
    match self {
      None => Some(0u),
      Some(_) => Some(1u)
    }
}
