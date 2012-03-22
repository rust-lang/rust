fn main() {

    let mut y: int = 42;
    let mut z: int = 42;
    let mut x: int;
    while z < 50 {
        z += 1;
        while false { x <- y; y = z; }
        log(debug, y);
    }
    assert (y == 42 && z == 50);
}
