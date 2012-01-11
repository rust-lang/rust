resource finish<T>(arg: {val: T, fin: native fn(T)}) {
    arg.fin(arg.val);
}

fn main() {
    let box = @mutable 10;
    fn dec_box(&&i: @mutable int) { *i -= 1; }

    { let i <- finish({val: box, fin: dec_box}); }
    assert (*box == 9);
}
