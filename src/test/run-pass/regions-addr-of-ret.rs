// xfail-test

fn f(x : &a.int) -> &a.int {
    ret &*x;
}

fn main() {
    log(error, #fmt("%d", *f(&3)));
}

