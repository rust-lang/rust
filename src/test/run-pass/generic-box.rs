

fn box<T: copy>(x: {x: T, y: T, z: T}) -> @{x: T, y: T, z: T} { ret @x; }

fn main() {
    let x: @{x: int, y: int, z: int} = box::<int>({x: 1, y: 2, z: 3});
    assert (x.y == 2);
}
