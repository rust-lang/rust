// error-pattern:refutable pattern

enum xx { xx(int), yy, }

fn main() {
    let @{x: xx(x), y: y} = @{x: xx(10), y: 20};
    assert (x + y == 30);
}
