// xfail-stage0
fn main() {

    let y: int = 42;
    let z: int = 42;
    let x: int;
    while z < 50 { z += 1; while false { x <- y; y = z; } log y; }
    assert (y == 42 && z == 50);
}