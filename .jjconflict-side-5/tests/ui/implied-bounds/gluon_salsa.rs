//@ check-pass
// Related to Bevy regression #115559, found in
// a crater run on #118553.

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
        panic!();
    }
}

fn main() {}
