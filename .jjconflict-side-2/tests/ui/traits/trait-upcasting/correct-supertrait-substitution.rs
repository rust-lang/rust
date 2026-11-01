//@ run-pass

trait Foo<T: Default + ToString>: Bar<i32> + Bar<T> {}
trait Bar<T: Default + ToString> {
    fn bar(&self) -> String {
        T::default().to_string()
    }
}

struct S1;

impl Bar<i32> for S1 {}
impl Foo<i32> for S1 {}

struct S2;
impl Bar<i32> for S2 {}
impl Bar<bool> for S2 {}
impl Foo<bool> for S2 {}

fn test1(x: &dyn Foo<i32>) {
    let s = x as &dyn Bar<i32>;
    assert_eq!("0", &s.bar().to_string());
}

fn test2(x: &dyn Foo<bool>) {
    let p = x as &dyn Bar<i32>;
    assert_eq!("0", &p.bar().to_string());
    let q = x as &dyn Bar<bool>;
    assert_eq!("false", &q.bar().to_string());
}

fn main() {
    let s1 = S1;
    test1(&s1);
    let s2 = S2;
    test2(&s2);
}
