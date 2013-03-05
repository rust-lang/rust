fn f(_: &const [int]) {}

fn main() {
    let v = [ 1, 2, 3 ];
    f(v);
}

