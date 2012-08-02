fn f(x : &a/int) -> &a/int {
    return &*x;
}

fn main() {
    let three = &3;
    log(error, fmt!{"%d", *f(three)});
}

