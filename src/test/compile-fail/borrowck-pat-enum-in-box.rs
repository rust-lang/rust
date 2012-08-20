fn match_imm_box(v: &const @Option<int>) -> int {
    match *v {
      @Some(ref i) => {*i}
      @None => {0}
    }
}

fn match_const_box(v: &const @const Option<int>) -> int {
    match *v {
      @Some(ref i) => { *i } // ok because this is pure
      @None => {0}
    }
}

pure fn pure_process(_i: int) {}

fn match_const_box_and_do_pure_things(v: &const @const Option<int>) {
    match *v {
      @Some(ref i) => {
        pure_process(*i)
      }
      @None => {}
    }
}

fn process(_i: int) {}

fn match_const_box_and_do_bad_things(v: &const @const Option<int>) {
    match *v {
      @Some(ref i) => { //~ ERROR illegal borrow unless pure
        process(*i) //~ NOTE impure due to access to impure function
      }
      @None => {}
    }
}

fn main() {
}
