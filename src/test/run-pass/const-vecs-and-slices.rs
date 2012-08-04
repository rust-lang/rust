const x : [int]/4 = [1,2,3,4];

fn main() {
    io::println(fmt!("%?", x[1]));
    assert x[1] == 2;
    assert x[3] == 4;
}