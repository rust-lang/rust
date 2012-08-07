fn process<T>(_t: T) {}

fn match_const_opt_by_mut_ref(v: &const option<int>) {
    match *v {
      some(ref mut i) => process(i), //~ ERROR illegal borrow
      none => ()
    }
}

fn match_const_opt_by_const_ref(v: &const option<int>) {
    match *v {
      some(ref const i) => process(i), //~ ERROR illegal borrow unless pure
      //~^ NOTE impure due to
      none => ()
    }
}

fn match_const_opt_by_imm_ref(v: &const option<int>) {
    match *v {
      some(ref i) => process(i), //~ ERROR illegal borrow unless pure
      //~^ NOTE impure due to
      none => ()
    }
}

fn match_const_opt_by_value(v: &const option<int>) {
    match *v {
      some(copy i) => process(i),
      none => ()
    }
}

fn main() {
}
