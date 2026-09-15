//@ revisions: current next
//@[current] compile-flags: -Znext-solver=no
//@[next] compile-flags: -Znext-solver=globally

trait Gat
where
    for<'a, 'b> Self::Assoc<'a, 'b>: 'a,
{
    type Assoc<'a, 'b>;

    fn assoc<'a, 'b>(first: &'a (), second: &'b ()) -> Self::Assoc<'a, 'b>;
}

fn gat_keeps_first_region_live<T: Gat>() {
    let second = ();
    let value;
    {
        let first = ();
        value = T::assoc(&first, &second);
        //~^ ERROR `first` does not live long enough
    }
    // The GAT's item bound becomes its first identity region, not 'static.
    drop(value);
}

fn main() {}
