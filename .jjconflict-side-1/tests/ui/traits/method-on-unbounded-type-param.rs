fn f<T>(a: T, b: T) -> std::cmp::Ordering {
    a.cmp(&b) //~ ERROR E0599
}
fn g<T>(a: T, b: T) -> std::cmp::Ordering {
    (&a).cmp(&b) //~ ERROR E0599
}
fn h<T>(a: &T, b: T) -> std::cmp::Ordering {
    a.cmp(&b) //~ ERROR E0599
}
trait T {}
impl<X: std::cmp::Ord> T for X {}
fn main() {
    let x: Box<dyn T> = Box::new(0);
    x.cmp(&x); //~ ERROR E0599
}
