// xfail-fast  (compile-flags unsupported on windows)
// compile-flags:--borrowck=err

fn match_imm_box(v: &const @option<int>) -> int {
    alt *v {
      @some(i) {i}
      @none {0}
    }
}

fn match_const_box(v: &const @const option<int>) -> int {
    alt *v {
      @some(i) {
        //!^ ERROR enum variant in aliasable, mutable location
        i
      }
      @none {0}
    }
}

fn main() {
}
