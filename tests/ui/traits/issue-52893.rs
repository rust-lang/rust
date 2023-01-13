// check-fail
//
// regression test for issue 52893
trait At<Name> {
    type AtRes;
    fn at(self) -> Self::AtRes;
}

trait Push<T> {
    type PushRes;
    fn push(self, other: T) -> Self::PushRes;
}

trait AddClass<Name, F> {
    type AddRes;
    fn init(self, func: F);
}

trait ToRef {
    type RefRes;
    fn to_ref(&self) -> Self::RefRes;
}

struct Class<P>(P);

impl<P> Class<P> {
    fn with<Name, F>(self) -> <Self as AddClass<Name, F>>::AddRes
    where
        Self: AddClass<Name, F>,
    {
        todo!()
    }

    fn from<F>(self) -> <Self as AddClass<P, F>>::AddRes
    where
        Self: AddClass<P, F>,
    {
        todo!()
    }
}

impl<F, Name, P> AddClass<Name, F> for Class<P>
where
    Self: At<Name>,
    <Self as At<Name>>::AtRes: Push<F>,
    <<Self as At<Name>>::AtRes as Push<F>>::PushRes: ToRef<RefRes = Self> + Push<F>,
{
    type AddRes = ();

    fn init(self, func: F) {
        let builder = self.at().push(func);
        let output = builder.to_ref();
        builder.push(output); //~ ERROR mismatched types [E0308]
    }
}

fn main() {}
