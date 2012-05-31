// error-pattern:moving out of immutable upvar
fn test(-x: uint) {}

fn main() {
    let i = 3u;
    for uint::range(0u, 10u) {|_x| test(i)}
}