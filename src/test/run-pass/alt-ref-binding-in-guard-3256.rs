fn main() {
    let x = Some(unsafe::exclusive(true));
    match move x {
        Some(ref z) if z.with(|b| *b) => {
            do z.with |b| { assert *b; }
        },
        _ => fail
    }
}