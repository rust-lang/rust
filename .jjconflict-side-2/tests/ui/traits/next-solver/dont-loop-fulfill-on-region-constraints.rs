//@ compile-flags: -Znext-solver
//@ check-pass

trait Eq<'a, 'b, T> {}

trait Ambig {}
impl Ambig for () {}

impl<'a, T> Eq<'a, 'a, T> for () where T: Ambig {}

fn eq<'a, 'b, T>(t: T)
where
    (): Eq<'a, 'b, T>,
{
}

fn test<'r>() {
    let mut x = Default::default();

    // When we evaluate `(): Eq<'r, 'r, ?0>` we uniquify the regions.
    // That leads us to evaluate `(): Eq<'?0, '?1, ?0>`. The response of this
    // will be ambiguous (because `?0: Ambig` is ambig) and also not an "identity"
    // response, since the region constraints will contain `'?0 == '?1` (so
    // `is_changed` will return true). Since it's both ambig and changed,
    // fulfillment will both re-register the goal AND loop again. This hits the
    // overflow limit. This should neither be considered overflow, nor ICE.
    eq::<'r, 'r, _>(x);

    x = ();
}

fn main() {}
