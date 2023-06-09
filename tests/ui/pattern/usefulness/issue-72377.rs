#[derive(PartialEq, Eq)]
enum X { A, B, C, }

fn main() {
    let x = X::A;
    let y = Some(X::A);

    match (x, y) {
        //~^ ERROR non-exhaustive patterns: `(X::A, Some(X::A))`, `(X::A, Some(X::B))`, `(X::B, Some(X::B))` and 2
        //~| more not covered
        (_, None) => false,
        (v, Some(w)) if v == w => true,
        (X::B, Some(X::C)) => false,
        (X::B, Some(X::A)) => false,
        (X::A, Some(X::C)) | (X::C, Some(X::A)) => false,
    };
}
