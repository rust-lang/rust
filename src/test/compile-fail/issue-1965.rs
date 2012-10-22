// error-pattern:moving out of captured outer immutable variable in a stack closure
fn test(-x: uint) {}

fn main() {
    let i = 3;
    for uint::range(0, 10) |_x| {test(move i)}
}
