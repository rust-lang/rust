//@ check-pass

use std::marker::PhantomData;

trait Lt<'a> {
    type T;
}
struct Id<T>(PhantomData<T>);
impl<'a,T> Lt<'a> for Id<T> {
    type T = T;
}

struct Ref<T>(PhantomData<T>) where T: ?Sized;
impl<'a,T> Lt<'a> for Ref<T>
where T: 'a + Lt<'a> + ?Sized
{
    type T = &'a T;
}
struct Mut<T>(PhantomData<T>) where T: ?Sized;
impl<'a,T> Lt<'a> for Mut<T>
where T: 'a + Lt<'a> + ?Sized
{
    type T = &'a mut T;
}

struct C<I,O>(for<'a> fn(<I as Lt<'a>>::T) -> O) where I: for<'a> Lt<'a>;


fn main() {
    let c = C::<Id<_>,_>(|()| 3);
    c.0(());

}
