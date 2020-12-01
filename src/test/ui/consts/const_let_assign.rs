// check-pass

struct S(i32);

const A: () = {
    let mut s = S(0);
    s.0 = 1;
};

fn main() {}
