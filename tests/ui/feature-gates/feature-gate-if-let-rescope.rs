// This test shows the code that could have been accepted by enabling #![feature(if_let_rescope)]

struct A;
struct B<'a, T>(&'a mut T);

impl A {
    fn f(&mut self) -> Option<B<'_, Self>> {
        Some(B(self))
    }
}

impl<'a, T> Drop for B<'a, T> {
    fn drop(&mut self) {
        // this is needed to keep NLL's hands off and to ensure
        // the inner mutable borrow stays alive
    }
}

fn main() {
    let mut a = A;
    if let None = a.f().as_ref() {
        unreachable!()
    } else {
        a.f().unwrap();
        //~^ ERROR cannot borrow `a` as mutable more than once at a time
    };
}
