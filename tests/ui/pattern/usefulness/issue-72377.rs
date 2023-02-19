#[derive(PartialEq, Eq)]
enum X { A, B, C, }

fn main() {
    let x = X::A;
    let y = Some(X::A);

    match (x, y) {
        //~^ ERROR match is non-exhaustive
        //~| more not covered
        (_, None) => false,
        (v, Some(w)) if v == w => true,
        (X::B, Some(X::C)) => false,
        (X::B, Some(X::A)) => false,
        (X::A, Some(X::C)) | (X::C, Some(X::A)) => false,
    };
}
