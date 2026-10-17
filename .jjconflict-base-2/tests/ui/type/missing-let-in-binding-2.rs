//@ run-rustfix

fn main() {
    _v: Vec<i32> = vec![1, 2, 3]; //~ ERROR expected identifier, found `:`
}
