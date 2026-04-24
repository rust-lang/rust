#![feature(fn_delegation)]

trait Bound<T> {}

trait Trait<'a, T, const X: usize>
where
    Self: Bound<T>,
{
    fn static_method<'c: 'c, U, const B: bool>(x: usize) {}
}

impl<'a, T, const X: usize> Trait<'a, T, X> for usize {}
impl<T> Bound<T> for usize {}

reuse <usize as Trait<'static, i32, 123>>::static_method as foo { self + 2 }
reuse <usize as Trait<'static, i32, 123>>::static_method::<'static, String, false> as foo2;
reuse <usize as Trait>::static_method as bar { self + 1 }
reuse <usize as Trait>::static_method::<'static, Vec<i32>, false> as bar2;

reuse Trait::static_method as error { self - 123 }
//~^ ERROR: type annotations needed
reuse Trait::<'static, i32, 123>::static_method as error2;
//~^ ERROR: type annotations needed
reuse Trait::<'static, i32, 123>::static_method::<'static, String, false> as error3;
//~^ ERROR: type annotations needed
reuse Trait::static_method::<'static, Vec<i32>, false> as error4 { self + 4 }
//~^ ERROR: type annotations needed

reuse <String as Trait>::static_method as error5;
//~^ ERROR: the trait bound `String: Trait<'a, T, X>` is not satisfied

struct Struct;
impl<'a, T, const X: usize> Trait<'a, T, X> for Struct {}
//~^ ERROR: the trait bound `Struct: Bound<T>` is not satisfied

reuse <Struct as Trait>::static_method as error6;
//~^ ERROR: the trait bound `Struct: Bound<T>` is not satisfied

fn main() {
    foo::<'static, String, true>(123);
    foo2(123);

    bar::<'static, 'static, i32, 123, String, false>(123);
    bar2::<'static, usize, 321>(123);
}
