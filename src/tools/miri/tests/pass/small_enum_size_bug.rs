#[allow(dead_code)]
enum E {
    A = 1,
    B = 2,
    C = 3,
}

fn main() {
    let enone = None::<E>;
    if let Some(..) = enone {
        panic!();
    }
}
