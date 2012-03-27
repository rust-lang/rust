enum taggy {
    cons(@mut taggy),
    nil,
}

fn f() {
    let box = @mut nil;
    *box = cons(box);
}

fn main() {
    f();
}

