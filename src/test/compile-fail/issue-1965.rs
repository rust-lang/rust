// error-pattern:tried to deinitialize a variable declared in a different
fn test(-x: uint) {}

fn main() {
    let i = 3u;
    for uint::range(0u, 10u) {|_x| test(i)}
}