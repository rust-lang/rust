//@ run-pass
type Array = [(); ((1 < 2) == false) as usize];

fn main() {
    let _: Array = [];
}
