type IMPL_T<A> = dvec::dvec<A>;

#[doc = "
Iterates through the current contents.

Attempts to access this dvec during iteration will fail.
"]
fn EACH<A>(self: IMPL_T<A>, f: fn(A) -> bool) {
    import dvec::extensions;
    self.swap({ |v| vec::each(v, f); v })
}

fn SIZE_HINT<A>(self: IMPL_T<A>) -> option<uint> {
    import dvec::extensions;
    some(self.len())
}
