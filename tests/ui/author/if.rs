#[allow(clippy::all)]

fn main() {
    #[clippy::author]
    let _ = if true {
        1 == 1;
    } else {
        2 == 2;
    };
}
