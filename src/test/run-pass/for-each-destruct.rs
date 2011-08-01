iter x() -> {x: int, y: int} {
    let i = 0;
    while i < 40 {
        put {x: i, y: 30 - i};
        i += 10;
    }
}

fn main() {
    for each {x, y}: {x: int, y: int} in x() {
        assert x + y == 30;
    }
}
