#[no_std];

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
    contents: int,
}

impl Trait<int> for S2 {
    fn new<U>(x: int, _: U) -> S2 {
        S2 {
            contents: x,
        }
    }
}

fn foo<'a>() {
    let _ = S::new::<int,f64>(1, 1.0);    //~ ERROR the impl referenced by this path has 1 type parameter, but 0 type parameters were supplied
    let _ = S::<'a,int>::new::<f64>(1, 1.0);  //~ ERROR expected 0 lifetime parameter(s)
    let _: S2 = Trait::new::<int,f64>(1, 1.0);    //~ ERROR the trait referenced by this path has 1 type parameter, but 0 type parameters were supplied
    let _: S2 = Trait::<'a,int>::new::<f64>(1, 1.0);   //~ ERROR expected 0 lifetime parameter(s)
}

fn main() {}
