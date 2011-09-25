
fn main() {
    let x = ~{x: 1};
    let bar = x;
    assert bar.x == 1;
}
