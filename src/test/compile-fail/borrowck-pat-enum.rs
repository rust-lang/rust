fn match_ref(&&v: Option<int>) -> int {
    match v {
      Some(ref i) => {
        *i
      }
      None => {0}
    }
}

fn match_ref_unused(&&v: Option<int>) {
    match v {
      Some(_) => {}
      None => {}
    }
}

fn match_const_reg(v: &const Option<int>) -> int {
    match *v {
      Some(ref i) => {*i} // OK because this is pure
      None => {0}
    }
}

fn impure(_i: int) {
}

fn match_const_reg_unused(v: &const Option<int>) {
    match *v {
      Some(_) => {impure(0)} // OK because nothing is captured
      None => {}
    }
}

fn match_const_reg_impure(v: &const Option<int>) {
    match *v {
      Some(ref i) => {impure(*i)} //~ ERROR illegal borrow unless pure
      //~^ NOTE impure due to access to impure function
      None => {}
    }
}

fn match_imm_reg(v: &Option<int>) {
    match *v {
      Some(ref i) => {impure(*i)} // OK because immutable
      None => {}
    }
}

fn main() {
}
