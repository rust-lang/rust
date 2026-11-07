//@ check-fail

trait Trait { type Assoc; }
impl<'a> Trait for &'a () { type Assoc = &'a (); }

struct MyTuple<T>(T);
impl MyTuple<<&'static () as Trait>::Assoc> {
    fn test(x: &(), y: &()) {
        Self(x);
        //~^ ERROR
        let _: Self = MyTuple(y);
        //~^ ERROR
    }
}

struct MyStruct<T> { val: T, }
impl MyStruct<<&'static () as Trait>::Assoc> {
    fn test(x: &(), y: &()) {
        Self { val: x };
        //~^ ERROR
        let _: Self = MyStruct { val: y };
        //~^ ERROR
    }
}

fn main() {}
