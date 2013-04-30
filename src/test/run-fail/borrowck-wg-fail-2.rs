// error-pattern:borrowed

struct S {
    x: int
}

fn main() {
    let x = @mut S { x: 3 };
    let y: &S = x;
    let z = x;
    z.x = 5;
}
