fn f(x : &a/int) -> &a/int {
    ret &*x;
}

fn main() {
    let three = &3;
    log(error, fmt!{"%d", *f(three)});
}

