// compile-flags:--borrowck=err

fn match_ref(&&v: option<int>) -> int {
    alt v {
      some(i) {
        //^ ERROR enum variant in aliasable, mutable location
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
      some(i) {
        //!^ ERROR enum variant in aliasable, mutable location
        i
      }
      none {0}
    }
}

fn match_const_reg_unused(v: &const option<int>) {
    alt *v {
      some(_) {}
      none {}
    }
}

fn match_imm_reg(v: &option<int>) -> int {
    alt *v {
      some(i) {i}
      none {0}
    }
}

fn main() {
}
