// error-pattern:moving out of captured outer immutable variable in a stack closure
fn test(-x: uint) {}

fn main() {
    let i = 3u;
    for uint::range(0u, 10u) |_x| {test(i)}
}
