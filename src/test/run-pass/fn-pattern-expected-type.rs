fn main() {
    let f: fn((int,int)) = |(x, y)| {
        assert x == 1;
        assert y == 2;
    };
    f((1, 2));
}

