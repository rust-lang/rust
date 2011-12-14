use std;
import uint;

fn test(x: bool, foo: @{x: int, y: int, z: int}) -> int {
    let bar = foo;
    let y: @{x: int, y: int, z: int};
    if x { y <- bar; } else { y = @{x: 4, y: 5, z: 6}; }
    ret y.y;
}

fn main() {
    let x = @{x: 1, y: 2, z: 3};
    uint::range(0u, 10000u) {|i|
        assert (test(true, x) == 2);
    }
    assert (test(false, x) == 5);
}
