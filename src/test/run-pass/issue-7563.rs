trait IDummy {
    fn do_nothing(&self);
}

struct A { a: int }
struct B<'self> { b: int, pa: &'self A }

    impl IDummy for A {
        fn do_nothing(&self) {
            println("A::do_nothing() is called");
        }
    }

impl<'self> B<'self> {
    fn get_pa(&self) -> &'self IDummy { self.pa as &'self IDummy }
}

pub fn main() {
    let sa = A { a: 100 };
    let sb = B { b: 200, pa: &sa };

    debug!("sa is %?", sa);
    debug!("sb is %?", sb);
    debug!("sb.pa is %?", sb.get_pa());
}
