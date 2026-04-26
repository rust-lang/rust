#![feature(fn_delegation)]

trait Bound<T> {}

trait Trait<'a, T, const X: usize>
where
    Self: Bound<T> + Sized,
{
    fn method<U, const B: bool>(&self, x: U) {}
    fn method2<U, const B: bool>(self, x: U) {}
    fn method3<U, const B: bool>(&mut self, x: U) {}
}

impl<'a, T, const X: usize> Trait<'a, T, X> for usize {}
impl<T> Bound<T> for usize {}

reuse <usize as Trait<'static, i32, 123>>::method as foo;
reuse <usize as Trait<'static, i32, 123>>::method::<String, false> as foo2;
reuse <usize as Trait>::method as bar;
reuse <usize as Trait>::method::<Vec<i32>, false> as bar2;

reuse Trait::<'static, i32, 123>::method as foo3;
reuse Trait::<'static, i32, 123>::method2::<String, false> as foo4;
reuse Trait::method3 as bar3;
reuse Trait::method3::<Vec<i32>, false> as bar4;

reuse <String as Trait>::method as error5;
//~^ ERROR: the trait bound `String: Trait<'a, T, X>` is not satisfied

struct Struct;
impl<'a, T, const X: usize> Trait<'a, T, X> for Struct {}
//~^ ERROR: the trait bound `Struct: Bound<T>` is not satisfied

reuse <Struct as Trait>::method as error6;
//~^ ERROR: the trait bound `Struct: Bound<T>` is not satisfied

fn main() {
    foo::<&'static str, true>(&123, "");
    foo2(&123, "".to_string());
    bar::<'static, i32, 123, String, false>(&123, "".to_string());
    bar2::<'static, usize, 321>(&123, vec![213, 123]);

    foo3::<String, true>(&123, "".to_string());
    //~^ ERROR: function takes 3 generic arguments but 2 generic arguments were supplied
    foo4(123, "".to_string());
    bar3::<'static, i32, 123, bool, false>(&mut 123, true);
    //~^ ERROR: function takes 5 generic arguments but 4 generic arguments were supplied
    bar4::<'static, usize, 321>(&mut 123, vec![123]);
    //~^ ERROR: function takes 3 generic arguments but 2 generic arguments were supplied

    foo3::<usize, String, true>(&123, "".to_string());
    foo4::<usize>(123, "".to_string());
    bar3::<'static, usize, i32, 123, String, false>(&mut 123, "".to_string());
    bar4::<'static, usize, usize, 321>(&mut 123, vec![123]);
}
