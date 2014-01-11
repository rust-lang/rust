trait IDummy {
    fn do_nothing(&self);
}

struct A { a: int }
struct B<'a> { b: int, pa: &'a A }

    impl IDummy for A {
        fn do_nothing(&self) {
            println!("A::do_nothing() is called");
        }
    }

impl<'a> B<'a> {
    fn get_pa(&self) -> &'a IDummy { self.pa as &'a IDummy }
}

pub fn main() {
    let sa = A { a: 100 };
    let sb = B { b: 200, pa: &sa };

    debug!("sa is {:?}", sa);
    debug!("sb is {:?}", sb);
    debug!("sb.pa is {:?}", sb.get_pa());
}
