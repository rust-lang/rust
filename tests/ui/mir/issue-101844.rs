//@ check-pass

pub trait FirstTrait {
    type Item;
    type Extra: Extra<(), Error = Self::Item>;
}

trait SecondTrait {
    type Item2;
}

trait ThirdTrait: SecondTrait {
    type Item3;
}

trait FourthTrait {
    type Item4;
}

impl<First> SecondTrait for First
where
    First: FirstTrait,
{
    type Item2 = First::Extra;
}

impl<Second, T> ThirdTrait for Second
where
    Second: SecondTrait<Item2 = T>,
{
    type Item3 = T;
}

impl<S, Third: ?Sized> FourthTrait for Third
where
    Third: ThirdTrait<Item3 = S>,
{
    type Item4 = S;
}

pub trait Extra<Request> {
    type Error;
}

struct ImplShoulExist<D, Req> {
    _gen: (D, Req),
}

impl<D, Req> ImplShoulExist<D, Req>
where
    D: FourthTrait,
    D::Item4: Extra<Req>,
    <D::Item4 as Extra<Req>>::Error: Into<()>,
{
    fn access_fn(_: D) {
        todo!()
    }
}

impl<D, Req> Extra<Req> for ImplShoulExist<D, Req> {
    type Error = ();
}

pub fn broken<MS>(ms: MS)
where
    MS: FirstTrait,
    MS::Item: Into<()>,
{
    // Error: Apparently Balance::new doesn't exist during MIR validation
    ImplShoulExist::<MS, ()>::access_fn(ms);
}

fn main() {}
