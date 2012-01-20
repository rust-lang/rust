fn test1() {
    enum bar { u(~int), w(int), }

    let x = u(~10);
    assert alt x {
      u(a) {
        log(error, a);
        *a
      }
      _ { 66 }
    } == 10;
}

fn main() {
    test1();
}
