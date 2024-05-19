mod a {
    struct S;
    impl S { }
}

fn foo(_: a::S) { //~ ERROR: struct `S` is private
}

fn main() {}
