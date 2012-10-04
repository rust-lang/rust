#[allow(non_camel_case_types)]
pub type IMPL_T<A> = dvec::DVec<A>;

/**
 * Iterates through the current contents.
 *
 * Attempts to access this dvec during iteration will fail.
 */
pub pure fn EACH<A>(self: &IMPL_T<A>, f: fn(v: &A) -> bool) {
    unsafe {
        do self.swap |v| {
            v.each(f);
            move v
        }
    }
}

pub pure fn SIZE_HINT<A>(self: &IMPL_T<A>) -> Option<uint> {
    Some(self.len())
}
