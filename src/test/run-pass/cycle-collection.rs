enum taggy {
    cons(@mutable taggy),
    nil,
}

fn f() {
    let box = @mutable nil;
    *box = cons(box);
}

fn main() {
    f();
}

