#![feature(ptr_metadata)]

use std::field::Field; //~ ERROR: use of unstable library feature `field_projections` [E0658]
use std::ops::place::Subplace; //~ ERROR: use of unstable library feature `field_projections` [E0658]
use std::field::field_of; //~ ERROR: use of unstable library feature `field_projections` [E0658]
use std::ops::place::DropHusk; //~ ERROR: use of unstable library feature `field_projections` [E0658]
use std::ops::place::PlaceProxy; //~ ERROR: use of unstable library feature `field_projections` [E0658]
use std::ops::place::BorrowPlace; //~ ERROR: use of unstable library feature `field_projections` [E0658]
use std::ops::place::NestPlace; //~ ERROR: use of unstable library feature `field_projections` [E0658]
use std::ops::place::DropPlace; //~ ERROR: use of unstable library feature `field_projections` [E0658]
use std::ops::place::MovePlace; //~ ERROR: use of unstable library feature `field_projections` [E0658]
use std::ops::place::ReadPlace; //~ ERROR: use of unstable library feature `field_projections` [E0658]
use std::ops::place::WrapPlace; //~ ERROR: use of unstable library feature `field_projections` [E0658]
use std::ops::place::WritePlace; //~ ERROR: use of unstable library feature `field_projections` [E0658]
use std::ptr;
use std::ptr::Pointee;

fn project_ref<F: Field>(
    //~^ ERROR: use of unstable library feature `field_projections` [E0658]
    r: &F::Base, //~ ERROR: use of unstable library feature `field_projections` [E0658]
) -> &F::Type
//~^ ERROR: use of unstable library feature `field_projections` [E0658]
where
    F::Type: Sized, //~ ERROR: use of unstable library feature `field_projections` [E0658]
{
    unsafe { &*ptr::from_ref(r).byte_add(F::OFFSET).cast() } //~ ERROR: use of unstable library feature `field_projections` [E0658]
}

struct MyPtr<T: ?Sized>(*mut T);

impl<T: ?Sized> PlaceProxy for MyPtr<T> {
    //~^ ERROR: use of unstable library feature `field_projections` [E0658]
    type Target = T; //~ ERROR: use of unstable library feature `field_projections` [E0658]
}

unsafe impl<T, S> ReadPlace<S> for MyPtr<T>
//~^ ERROR: use of unstable library feature `field_projections` [E0658]
where
    T: ?Sized,
    S: Subplace<Source = T>, //~ ERROR: use of unstable library feature `field_projections` [E0658]
    //~^ ERROR: use of unstable library feature `field_projections` [E0658]
    //~^^ ERROR: use of unstable library feature `field_projections` [E0658]
    S::Target: Sized, //~ ERROR: use of unstable library feature `field_projections` [E0658]
{
    const SAFETY: bool = true; //~ ERROR: use of unstable library feature `field_projections` [E0658]

    unsafe fn read(this: *const Self, sub: S) -> S::Target {
        //~^ ERROR: use of unstable library feature `field_projections` [E0658]
        //~^^ ERROR: use of unstable library feature `field_projections` [E0658]
        let _ = (this, sub);
        todo!()
    }
}

unsafe impl<T, S> WritePlace<S> for MyPtr<T>
//~^ ERROR: use of unstable library feature `field_projections` [E0658]
where
    T: ?Sized,
    S: Subplace<Source = T>, //~ ERROR: use of unstable library feature `field_projections` [E0658]
    //~^ ERROR: use of unstable library feature `field_projections` [E0658]
    //~^^ ERROR: use of unstable library feature `field_projections` [E0658]
    S::Target: Sized, //~ ERROR: use of unstable library feature `field_projections` [E0658]
{
    const SAFETY: bool = true; //~ ERROR: use of unstable library feature `field_projections` [E0658]

    unsafe fn write(this: *const Self, sub: S, value: S::Target) {
        //~^ ERROR: use of unstable library feature `field_projections` [E0658]
        //~^^ ERROR: use of unstable library feature `field_projections` [E0658]
        let _ = (this, sub, value);
        todo!()
    }
}

unsafe impl<T, S> MovePlace<S> for MyPtr<T>
//~^ ERROR: use of unstable library feature `field_projections` [E0658]
where
    S: Subplace<Source = T>, //~ ERROR: use of unstable library feature `field_projections` [E0658]
    //~^ ERROR: use of unstable library feature `field_projections` [E0658]
    //~^^ ERROR: use of unstable library feature `field_projections` [E0658]
    S::Target: Sized, //~ ERROR: use of unstable library feature `field_projections` [E0658]
{
}

unsafe impl<T, S> DropPlace<S> for MyPtr<T>
//~^ ERROR: use of unstable library feature `field_projections` [E0658]
where
    T: ?Sized,
    S: Subplace<Source = T>, //~ ERROR: use of unstable library feature `field_projections` [E0658]
                             //~^ ERROR: use of unstable library feature `field_projections` [E0658]
{
    unsafe fn drop(this: *const Self, sub: S) {
        //~^ ERROR: use of unstable library feature `field_projections` [E0658]
        let _ = (this, sub);
        todo!()
    }
}

unsafe impl<T> DropHusk for MyPtr<T> {
    //~^ ERROR: use of unstable library feature `field_projections` [E0658]

    unsafe fn drop_husk(this: *const Self) {
        //~^ ERROR: use of unstable library feature `field_projections` [E0658]
        let _ = this;
        todo!()
    }
}

unsafe impl<T, S> BorrowPlace<S, MyPtr<S::Target>> for MyPtr<T>
//~^ ERROR: use of unstable library feature `field_projections` [E0658]
//~^^ ERROR: use of unstable library feature `field_projections` [E0658]
where
    T: ?Sized,
    S: Subplace<Source = T>, //~ ERROR: use of unstable library feature `field_projections` [E0658]
                             //~^ ERROR: use of unstable library feature `field_projections` [E0658]
                             //~^^ ERROR: use of unstable library feature `field_projections` [E0658]
{
    const SAFETY: bool = true; //~ ERROR: use of unstable library feature `field_projections` [E0658]

    unsafe fn borrow(this: *const Self, sub: S) -> MyPtr<S::Target> {
        //~^ ERROR: use of unstable library feature `field_projections` [E0658]
        //~^^ ERROR: use of unstable library feature `field_projections` [E0658]
        let _ = (this, sub);
        todo!()
    }
}

unsafe impl<T, S> NestPlace<S> for MyPtr<T>
//~^ ERROR: use of unstable library feature `field_projections` [E0658]
where
    S: Subplace<Source = T>, //~ ERROR: use of unstable library feature `field_projections` [E0658]
    //~^ ERROR: use of unstable library feature `field_projections` [E0658]
    //~^^ ERROR: use of unstable library feature `field_projections` [E0658]
    S::Target: PlaceProxy, //~ ERROR: use of unstable library feature `field_projections` [E0658]
                      //~^ ERROR: use of unstable library feature `field_projections` [E0658]
{
    unsafe fn nested(this: *const Self, sub: S) -> *const S::Target {
        //~^ ERROR: use of unstable library feature `field_projections` [E0658]
        //~^^ ERROR: use of unstable library feature `field_projections` [E0658]
        let _ = (this, sub);
        todo!()
    }
}

unsafe fn using_place_ops<T, S>(place: *const MyPtr<T>, sub: impl Fn() -> S)
where
    S: Subplace<Source = T>, //~ ERROR: use of unstable library feature `field_projections` [E0658]
    //~^ ERROR: use of unstable library feature `field_projections` [E0658]
    //~^^ ERROR: use of unstable library feature `field_projections` [E0658]
    S::Target: Sized + PlaceProxy, //~ ERROR: use of unstable library feature `field_projections` [E0658]
                              //~^ ERROR: use of unstable library feature `field_projections` [E0658]
{
    unsafe {
        let value = ReadPlace::read(place, sub()); //~ ERROR: use of unstable library feature `field_projections` [E0658]
        WritePlace::write(place, sub(), value); //~ ERROR: use of unstable library feature `field_projections` [E0658]
        let _ = BorrowPlace::<_, MyPtr<_>>::borrow(place, sub()); //~ ERROR: use of unstable library feature `field_projections` [E0658]
        let _ = NestPlace::nested(place, sub()); //~ ERROR: use of unstable library feature `field_projections` [E0658]
        DropPlace::drop(place, sub()); //~ ERROR: use of unstable library feature `field_projections` [E0658]
        DropHusk::drop_husk(place); //~ ERROR: use of unstable library feature `field_projections` [E0658]
    }
}

struct MyWrapper<T: ?Sized>(T);

impl<T: ?Sized> PlaceProxy for MyWrapper<T> {
    //~^ ERROR: use of unstable library feature `field_projections` [E0658]
    type Target = T; //~ ERROR: use of unstable library feature `field_projections` [E0658]
}

unsafe impl<T, S> WrapPlace<S> for MyWrapper<T>
//~^ ERROR: use of unstable library feature `field_projections` [E0658]
where
    T: ?Sized,
    S: Subplace<Source = T>, //~ ERROR: use of unstable library feature `field_projections` [E0658]
                             //~^ ERROR: use of unstable library feature `field_projections` [E0658]
{
    type Wrapped = MyWrapped<S>; //~ ERROR: use of unstable library feature `field_projections` [E0658]

    fn wrap(sub: S) -> MyWrapped<S> {
        //~^ ERROR: use of unstable library feature `field_projections` [E0658]
        MyWrapped(sub)
    }
}

#[derive(Copy, Clone)]
struct MyWrapped<S>(S);

unsafe impl<S> Subplace for MyWrapped<S>
//~^ ERROR: use of unstable library feature `field_projections` [E0658]
//~^^ ERROR: explicit impls for the `Subplace` trait are not permitted [E0322]
where
    S: Subplace, //~ ERROR: use of unstable library feature `field_projections` [E0658]
{
    type Source = MyWrapper<S::Source>; //~ ERROR: use of unstable library feature `field_projections` [E0658]
    //~^ ERROR: use of unstable library feature `field_projections` [E0658]
    type Target = MyWrapper<S::Target>; //~ ERROR: use of unstable library feature `field_projections` [E0658]
    //~^ ERROR: use of unstable library feature `field_projections` [E0658]

    fn offset(
        //~^ ERROR: use of unstable library feature `field_projections` [E0658]
        self,
        metadata: <Self::Source as Pointee>::Metadata,
        //~^ ERROR: use of unstable library feature `field_projections` [E0658]
    ) -> (usize, <Self::Target as Pointee>::Metadata) {
        //~^ ERROR: use of unstable library feature `field_projections` [E0658]
        let _ = (self, metadata);
        todo!()
    }
}

unsafe fn using_wrapper_ops<T, S>(sub: S)
where
    T: ?Sized,
    S: Subplace<Source = T>, //~ ERROR: use of unstable library feature `field_projections` [E0658]
    //~^ ERROR: use of unstable library feature `field_projections` [E0658]
{
    unsafe {
        let _ = <MyWrapper<T> as WrapPlace<S>>::wrap(sub); //~ ERROR: use of unstable library feature `field_projections` [E0658]
    }
}

fn main() {
    struct Foo(());
    let _ = project_ref::<field_of!(Foo, 0)>(&Foo(())); //~ ERROR: use of unstable library feature `field_projections` [E0658]
}
