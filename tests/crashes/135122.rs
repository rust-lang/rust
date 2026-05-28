//@ known-bug: #135122
trait Add {
    type Output;
    fn add(_: (), _: Self::Output) {}
}

trait IsSame<Lhs> {
    type Assoc;
}

trait Data {
    type Elem;
}

impl<B> IsSame<i16> for f32 where f32: IsSame<B, Assoc = B> {}

impl<A> Add for i64
where
    f32: IsSame<A>,
    i8: Data<Elem = A>,
{
    type Output = <f32 as IsSame<A>>::Assoc;
    fn add(_: Data, _: Self::Output) {}
}
