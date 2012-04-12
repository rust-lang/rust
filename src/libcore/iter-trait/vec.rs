type IMPL_T<A> = [const A];

fn EACH<A>(self: IMPL_T<A>, f: fn(A) -> bool) {
    vec::each(self, f)
}

fn SIZE_HINT<A>(self: IMPL_T<A>) -> option<uint> {
    some(vec::len(self))
}
