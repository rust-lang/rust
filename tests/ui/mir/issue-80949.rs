//@ build-pass

trait Trait { type Item; }

impl<'a, X> Trait for &'a Vec<X> {
    type Item = &'a X;
}

impl<X> Trait for Box<dyn Trait<Item = X>> {
    type Item = X;
}

fn make_dyn_trait(_: &()) -> Box<dyn Trait<Item = &()>> {
    todo!()
}

fn diff<'a, M, N, S>(_: N, _: S)
where
    M: 'a,
    N: Trait<Item = &'a M>,
    S: Trait<Item = &'a M>,
{
    todo!()
}

fn may_panic<X>(_: X) { }

fn main() {
    let dyn_trait = make_dyn_trait(&());
    let storage = vec![()];
    may_panic(());
    let storage_ref = &storage;
    diff(dyn_trait, storage_ref);
}
