// error-pattern:borrowed

struct S {
    x: int
}

fn main() {
    let x = @mut S { x: 3 };
    let y: &S = x;
    x.x = 5;
}
