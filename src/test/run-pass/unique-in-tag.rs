fn test1() {
    tag bar { u(~int); w(int); }

    let x = u(~10);
    assert alt x {
      u(a) {
        log_err a;
        *a
      }
      _ { 66 }
    } == 10;
}

fn main() {
    test1();
}
