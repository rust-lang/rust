export rsrc;

fn foo(_x: i32) {
}

resource rsrc(x: i32) {
    foo(x);
}