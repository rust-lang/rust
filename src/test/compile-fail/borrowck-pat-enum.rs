fn match_ref(&&v: option<int>) -> int {
    alt v {
      some(i) {
        i
      }
      none {0}
    }
}

fn match_ref_unused(&&v: option<int>) {
    alt v {
      some(_) {}
      none {}
    }
}

fn match_const_reg(v: &const option<int>) -> int {
    alt *v {
      some(i) {i} // OK because this is pure
      none {0}
    }
}

fn impure(_i: int) {
}

fn match_const_reg_unused(v: &const option<int>) {
    alt *v {
      some(_) {impure(0)} // OK because nothing is captured
      none {}
    }
}

fn match_const_reg_impure(v: &const option<int>) {
    alt *v {
      some(i) {impure(i)} //~ ERROR illegal borrow unless pure: enum variant in aliasable, mutable location
      //~^ NOTE impure due to access to impure function
      none {}
    }
}

fn match_imm_reg(v: &option<int>) {
    alt *v {
      some(i) {impure(i)} // OK because immutable
      none {}
    }
}

fn main() {
}
