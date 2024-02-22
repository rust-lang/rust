//@ revisions: rpass1 rpass2

trait MyTrait: One + Two {}
impl<T> One for T {
    fn method_one(&self) -> usize {
        1
    }
}
impl<T> Two for T {
    fn method_two(&self) -> usize {
        2
    }
}
impl<T: One + Two> MyTrait for T {}

fn main() {
    let a: &dyn MyTrait = &true;
    assert_eq!(a.method_one(), 1);
    assert_eq!(a.method_two(), 2);
}

// Re-order traits 'One' and 'Two' between compilation
// sessions

#[cfg(rpass1)]
trait One { fn method_one(&self) -> usize; }

trait Two { fn method_two(&self) -> usize; }

#[cfg(rpass2)]
trait One { fn method_one(&self) -> usize; }
