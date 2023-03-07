// run-pass
#![allow(dead_code)]
trait IDummy {
    fn do_nothing(&self);
}

#[derive(Debug)]
struct A { a: isize }
#[derive(Debug)]
struct B<'a> { b: isize, pa: &'a A }

    impl IDummy for A {
        fn do_nothing(&self) {
            println!("A::do_nothing() is called");
        }
    }

impl<'a> B<'a> {
    fn get_pa(&self) -> &'a dyn IDummy { self.pa as &'a dyn IDummy }
}

pub fn main() {
    let sa = A { a: 100 };
    let sb = B { b: 200, pa: &sa };

    println!("sa is {:?}", sa);
    println!("sb is {:?}", sb);
}
