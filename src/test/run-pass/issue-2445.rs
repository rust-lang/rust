import dvec::dvec;

class c1<T: copy> {
  let x: T;
  new(x: T) {self.x = x;}

    fn f1(x: T) {}
}

impl i1<T: copy> for c1<T> {
    fn f2(x: T) {}
}


fn main() {
    c1::<int>(3).f1(4);
    c1::<int>(3).f2(4);
}
