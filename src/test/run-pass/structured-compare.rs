

enum foo { large, small, }

fn main() {
    let a = {x: 1, y: 2, z: 3};
    let b = {x: 1, y: 2, z: 3};
    assert (a == b);
    assert (a != {x: 1, y: 2, z: 4});
    assert (a < {x: 1, y: 2, z: 4});
    assert (a <= {x: 1, y: 2, z: 4});
    assert ({x: 1, y: 2, z: 4} > a);
    assert ({x: 1, y: 2, z: 4} >= a);
    let x = large;
    let y = small;
    assert (x != y);
    assert (x == large);
    assert (x != small);
}
