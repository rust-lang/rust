struct Bar(int, int);

const X: Bar = Bar(1, 2);

fn main() {
    match X {
        Bar(x, y) => {
            assert x == 1;
            assert y == 2;
        }
    }
}

