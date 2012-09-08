use dvec::DVec;

struct c1<T: Copy> {
    x: T,
}

impl<T: Copy> c1<T> {
    fn f1(x: T) {}
}

fn c1<T: Copy>(x: T) -> c1<T> {
    c1 {
        x: x
    }
}

impl<T: Copy> c1<T> {
    fn f2(x: T) {}
}


fn main() {
    c1::<int>(3).f1(4);
    c1::<int>(3).f2(4);
}
