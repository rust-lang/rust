import test1::*;
import test2::*;

mod circ1 {
  import circ1::*;
  fn f1() {
    log "f1";
  }
  fn common() -> uint {
    ret 0u;
  }
}

mod circ2 {
  import circ2::*;
  fn f2() {
    log "f2";
  }
  fn common() -> uint {
    ret 1u;
  }
}

mod test1 {
  import circ1::*;
  fn test1() {
    f1();
    f2();
    assert(common() == 0u);
  }
}

mod test2 {
  import circ2::*;
  fn test2() {
    f1();
    f2();
    assert(common() == 1u);
  }
}



fn main() {
  test1();
  test2();
}