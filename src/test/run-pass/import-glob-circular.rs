import test1::*;
import test2::*;

mod circ1 {
  import circ1::*;
  export f1;
  export f2;
  export common;
  fn f1() -> uint {
    ret 1u
  }
  fn common() -> uint {
    ret 1u;
  }
}

mod circ2 {
  import circ2::*;
  export f1;
  export f2;
  export common;
  fn f2() -> uint {
    ret 2u;
  }
  fn common() -> uint {
    ret 2u;
  }
}

mod test1 {
  import circ1::*;
  fn test1() {
    assert(f1() == 1u);
    //make sure that cached lookups work...
    assert(f1() == 1u);
    assert(f2() == 2u);
    assert(f2() == 2u);
    assert(common() == 1u);
    assert(common() == 1u);
  }
}

mod test2 {
  import circ2::*;
  fn test2() {
    assert(f1() == 1u);
    //make sure that cached lookups work...
    assert(f1() == 1u);
    assert(f2() == 2u);
    assert(f2() == 2u);
    assert(common() == 2u);    
    assert(common() == 2u);
  }
}



fn main() {
  test1();
  test2();
}