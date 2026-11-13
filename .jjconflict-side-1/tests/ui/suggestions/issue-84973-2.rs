// A slight variation of issue-84973.rs. Here, a mutable borrow is
// required (and the obligation kind is different).

trait Tr {}
impl Tr for &mut i32 {}

fn foo<T: Tr>(i: T) {}

fn main() {
    let a: i32 = 32;
    foo(a);
    //~^ ERROR: the trait bound `i32: Tr` is not satisfied [E0277]
}
