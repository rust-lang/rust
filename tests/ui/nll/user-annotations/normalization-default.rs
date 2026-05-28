//@ check-fail

trait Trait { type Assoc; }
impl<'a> Trait for &'a () { type Assoc = &'a (); }

struct MyTuple<T, U = <&'static () as Trait>::Assoc>(T, U);
fn test_tuple(x: &(), y: &()) {
    MyTuple::<_>((), x);
    //~^ ERROR
    let _: MyTuple::<_> = MyTuple((), y);
    //~^ ERROR
}

struct MyStruct<T, U = <&'static () as Trait>::Assoc> { val: (T, U), }
fn test_struct(x: &(), y: &()) {
    MyStruct::<_> { val: ((), x) };
    //~^ ERROR
    let _: MyStruct::<_> = MyStruct { val: ((), y) };
    //~^ ERROR
}

fn main() {}
