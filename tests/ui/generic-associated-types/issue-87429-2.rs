// Derived from `issue-87429`. A test that ensures that using bound vars in the
// predicates in the param env when checking that an associated type satisfies
// its bounds does not cause us to not be able to use the bounds on the parameters.

//@ check-pass

trait Family {
    type Member<'a, C: Eq>: for<'b> MyBound<'b, C>;
}

trait MyBound<'a, C> { }
impl<'a, C: Eq> MyBound<'a, C> for i32 { }

impl Family for () {
    type Member<'a, C: Eq> = i32;
}

fn main() {}
