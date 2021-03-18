struct S<T> {
    contents: T,
}

impl<T> S<T> {
    fn new<U>(x: T, _: U) -> S<T> {
        S {
            contents: x,
        }
    }
}

trait Trait<T> {
    fn new<U>(x: T, y: U) -> Self;
}

struct S2 {
    contents: isize,
}

impl Trait<isize> for S2 {
    fn new<U>(x: isize, _: U) -> S2 {
        S2 {
            contents: x,
        }
    }
}

fn foo<'a>() {
    let _ = S::new::<isize,f64>(1, 1.0);
    //~^ ERROR this associated function takes 1 type argument but 2 type arguments were supplied

    let _ = S::<'a,isize>::new::<f64>(1, 1.0);
    //~^ ERROR this struct takes 0 lifetime arguments but 1 lifetime argument was supplied

    let _: S2 = Trait::new::<isize,f64>(1, 1.0);
    //~^ ERROR this associated function takes 1 type argument but 2 type arguments were supplied

    let _: S2 = Trait::<'a,isize>::new::<f64,f64>(1, 1.0);
    //~^ ERROR this trait takes 0 lifetime arguments but 1 lifetime argument was supplied
    //~| ERROR this associated function takes 1 type argument but 2 type arguments were supplied
}

fn main() {}
