//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally
//@ check-pass


// Regression test for https://github.com/rust-lang/rust/issues/166325
// Proving <<T as ConstructionFirm>::Builder as BuilderFn<'_>>::Output: '_ from its own
// implied bound used to ICE with debug asserts. Only the outer alias of the goal was made
// non rigid before matching it against the assumption, so the nested
// <T as ConstructionFirm>::Builder had a different rigidness and boom.

pub trait BuilderFn<'a> {
    type Output;
}

pub trait ConstructionFirm {
    type Builder: for<'a> BuilderFn<'a>;
}

pub trait Campus<T>
where
    T: ConstructionFirm,
{
    fn add_building(
        &mut self,
        building: &mut <<T as ConstructionFirm>::Builder as BuilderFn<'_>>::Output,
    );
}

fn main() {}
