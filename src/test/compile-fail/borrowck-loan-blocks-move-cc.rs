// compile-flags:--borrowck=err

fn borrow(v: &int, f: fn(x: &int)) {
    f(v);
}

fn box_imm() {
    let mut v = ~3;
    let _w = &mut v; //! NOTE loan of mutable local variable granted here
    task::spawn { |move v|
        //!^ ERROR moving out of mutable local variable prohibited due to outstanding loan
        #debug["v=%d", *v];
    }

    let mut v = ~3;
    let _w = &mut v; //! NOTE loan of mutable local variable granted here
    task::spawn(fn~(move v) {
        //!^ ERROR moving out of mutable local variable prohibited due to outstanding loan
        #debug["v=%d", *v];
    });
}

fn main() {
}
