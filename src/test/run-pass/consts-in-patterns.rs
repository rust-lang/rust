const FOO: int = 10;
const BAR: int = 3;

fn main() {
    let x: int = 3;
    let y = match x {
        FOO => 1,
        BAR => 2,
        _ => 3
    };
    assert y == 2;
}

