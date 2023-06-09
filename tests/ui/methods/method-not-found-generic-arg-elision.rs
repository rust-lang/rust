// Test for issue 81576
// Remove generic arguments if no method is found for all possible generic argument

use std::marker::PhantomData;

struct Wrapper2<'a, T, const C: usize> {
    x: &'a T,
}

impl<'a, const C: usize> Wrapper2<'a, i8, C> {
    fn method(&self) {}
}

impl<'a, const C: usize> Wrapper2<'a, i16, C> {
    fn method(&self) {}
}

impl<'a, const C: usize> Wrapper2<'a, i32, C> {
    fn method(&self) {}
}
struct Wrapper<T>(T);

impl Wrapper<i8> {
    fn method(&self) {}
}

impl Wrapper<i16> {
    fn method(&self) {}
}

impl Wrapper<i32> {
    fn method(&self) {}
}

impl Wrapper<i64> {
    fn method(&self) {}
}

impl Wrapper<u8> {
    fn method(&self) {}
}

impl Wrapper<u16> {
    fn method(&self) {}
}

struct Point<T> {
    x: T,
    y: T,
}

impl Point<f64> {
    fn distance(&self) -> f64 {
        self.x.hypot(self.y)
    }
}

struct Other;

impl Other {
    fn other(&self) {}
}

struct Struct<T> {
    _phatom: PhantomData<T>,
}

impl<T> Default for Struct<T> {
    fn default() -> Self {
        Self { _phatom: PhantomData }
    }
}

impl<T: Clone + Copy + PartialEq + Eq + PartialOrd + Ord> Struct<T> {
    fn method(&self) {}
}

fn main() {
    let point_f64 = Point { x: 1_f64, y: 1_f64 };
    let d = point_f64.distance();
    let point_i32 = Point { x: 1_i32, y: 1_i32 };
    let d = point_i32.distance();
    //~^ ERROR no method named `distance` found for struct `Point<i32>
    let d = point_i32.other();
    //~^ ERROR no method named `other` found for struct `Point
    let v = vec![1, 2, 3];
    v.iter().map(Box::new(|x| x * x) as Box<dyn Fn(&i32) -> i32>).extend(std::iter::once(100));
    //~^ ERROR no method named `extend` found for struct `Map
    let wrapper = Wrapper(true);
    wrapper.method();
    //~^ ERROR no method named `method` found for struct `Wrapper<bool>
    wrapper.other();
    //~^ ERROR no method named `other` found for struct `Wrapper
    let boolean = true;
    let wrapper = Wrapper2::<'_, _, 3> { x: &boolean };
    wrapper.method();
    //~^ ERROR no method named `method` found for struct `Wrapper2<'_, bool, 3>
    wrapper.other();
    //~^ ERROR no method named `other` found for struct `Wrapper2
    let a = vec![1, 2, 3];
    a.not_found();
    //~^ ERROR no method named `not_found` found for struct `Vec
    let s = Struct::<f64>::default();
    s.method();
    //~^ ERROR the method `method` exists for struct `Struct<f64>`, but its trait bounds were not satisfied
}
