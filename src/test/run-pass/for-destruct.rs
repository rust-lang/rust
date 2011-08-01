fn main() {
    for {x, y}: {x: int, y: int} in ~[{x: 10, y: 20}, {x: 30, y: 0}] {
        assert x + y == 30;
    }
}
