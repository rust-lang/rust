//@ revisions: current next
//@[current] compile-flags: -Znext-solver=no
//@[next] compile-flags: -Znext-solver=globally
//@ check-pass

trait Gat
where
    for<'a, 'b> Self::Assoc<'a, 'b>: 'a,
{
    type Assoc<'a, 'b>;

    fn assoc<'a, 'b>(first: &'a (), second: &'b ()) -> Self::Assoc<'a, 'b>;
}

fn gat_uses_identity_region<T: Gat>() {
    let first = ();
    let value;
    {
        let second = ();
        value = T::assoc(&first, &second);
    }
    // The item bound maps the GAT's bound regions to its identity parameters.
    // Only the first argument needs to stay live.
    drop(value);
}

trait Static<'env>
where
    for<'a> Self::Assoc: 'a,
{
    type Assoc;

    fn assoc(value: &'env ()) -> Self::Assoc;
}

fn remaining_bound_region_is_static<T: for<'env> Static<'env>>() {
    let value;
    {
        let local = ();
        value = T::assoc(&local);
    }
    // The bound region is absent from the alias's arguments, so the associated
    // type outlives every region, including 'static.
    drop(value);
}

fn main() {}
