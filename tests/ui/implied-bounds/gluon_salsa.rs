//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] check-pass

// Related to Bevy regression #115559, found in
// a crater run on #118553.
//
// Normalizing when computing implied bounds and normalizing the
// signature when checking that it is well-formed happen separately.
//
// In the old solver both use the projection_cache so the resulting type
// has the same unconstrained infer var.
//
// The new solver does not have any per-infcx caches, so the unconstrained
// variables differ, causing lexical region error to fail to prove the relevant
// type outlives.

pub trait QueryBase {
    type Db;
}

pub trait AsyncQueryFunction<'f>: // 'f is important
    QueryBase<Db = <Self as AsyncQueryFunction<'f>>::SendDb> // bound is important
{
    type SendDb;
}

pub struct QueryTable<'me, Q, DB> {
    _q: Option<Q>,
    _db: Option<DB>,
    _marker: Option<&'me ()>,
}

impl<'me, Q> QueryTable<'me, Q, <Q as QueryBase>::Db>
where
    Q: for<'f> AsyncQueryFunction<'f>,
{
    // When borrowchechking this function we normalize `<Q as QueryBase>::Db` in the
    // function signature to `<Self as QueryFunction<'?x>>::SendDb`, where `'?x` is an
    // unconstrained region variable. We then addd `<Self as QueryFunction<'?x>>::SendDb: 'a`
    // as an implied bound. We currently a structural equality to decide whether this bound
    // should be used to prove the bound  `<Self as QueryFunction<'?x>>::SendDb: 'a`. For this
    // to work we may have to structurally resolve regions as the actually used vars may
    // otherwise be semantically equal but structurally different.
    pub fn get_async<'a>(&'a mut self) {
        //[next]~^ ERROR: the associated type `<Q as AsyncQueryFunction<'_>>::SendDb` may not live long enough
        panic!();
    }
}

fn main() {}
