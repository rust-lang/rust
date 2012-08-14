#[allow(non_camel_case_types)]
type IMPL_T<A> = dvec::DVec<A>;

/**
 * Iterates through the current contents.
 *
 * Attempts to access this dvec during iteration will fail.
 */
fn EACH<A>(self: IMPL_T<A>, f: fn(A) -> bool) {
    self.swap(|v| { vec::each(v, f); v })
}

fn SIZE_HINT<A>(self: IMPL_T<A>) -> option<uint> {
    some(self.len())
}
