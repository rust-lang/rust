// xfail-boot
// xfail-stage0
// xfail-stage1
// xfail-stage2
iter x() -> int {
}

fn f() -> bool {
    for each (int i in x()) {
        ret true;
    }
    ret false;
}

fn main(vec[str] args) -> () {
  f();
}
