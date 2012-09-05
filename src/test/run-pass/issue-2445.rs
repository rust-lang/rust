use dvec::DVec;

struct c1<T: copy> {
  let x: T;
  new(x: T) {self.x = x;}

    fn f1(x: T) {}
}

impl<T: copy> c1<T> {
    fn f2(x: T) {}
}


fn main() {
    c1::<int>(3).f1(4);
    c1::<int>(3).f2(4);
}
