fn main() {
    enum t { t1(int), t2(int), }

    let x = ~t1(10);

    alt *x {
      t1(a) {
        assert a == 10;
      }
      _ { fail; }
    }

    alt x {
      ~t1(a) {
        assert a == 10;
      }
      _ { fail; }
    }
}