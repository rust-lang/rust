// compile-flags:--borrowck=err

fn borrow(v: &int, f: fn(x: &int)) {
    f(v);
}

fn box_imm() {
    let mut v = ~3;
    borrow(v) { |w| //! NOTE loan of mutable local variable granted here
        v = ~4; //! ERROR assigning to mutable upvar prohibited due to outstanding loan
        assert *v == 3;
        assert *w == 4;
    }
}

fn main() {
}
