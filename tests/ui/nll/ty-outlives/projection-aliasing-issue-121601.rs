//@ check-pass

pub trait Trait1 {
    type Output1;
    fn call<'z>(&'z self) -> &'z Self::Output1;
}

pub trait Trait2<T> {
    type Output2;
    fn call2<'x>(_: &'x T) -> &'x Self::Output2;
}

impl<A, B, T: Trait1<Output1 = A>> Trait2<T> for B
// Mind this `A` here
where
    B: Trait2<A>,
{
    type Output2 = <B as Trait2<A>>::Output2;
    fn call2<'y>(source: &'y T) -> &'y Self::Output2 {
        let t = source.call();
        B::call2(t)
    }
}

fn main() {}
