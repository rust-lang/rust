fn process<T>(_t: T) {}

fn match_const_opt_by_mut_ref(v: &const Option<int>) {
    match *v {
      Some(ref mut i) => process(i), //~ ERROR illegal borrow
      None => ()
    }
}

fn match_const_opt_by_const_ref(v: &const Option<int>) {
    match *v {
      Some(ref const i) => process(i), //~ ERROR illegal borrow unless pure
      //~^ NOTE impure due to
      None => ()
    }
}

fn match_const_opt_by_imm_ref(v: &const Option<int>) {
    match *v {
      Some(ref i) => process(i), //~ ERROR illegal borrow unless pure
      //~^ NOTE impure due to
      None => ()
    }
}

fn match_const_opt_by_value(v: &const Option<int>) {
    match *v {
      Some(copy i) => process(i),
      None => ()
    }
}

fn main() {
}
