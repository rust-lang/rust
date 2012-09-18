fn main() {
    let x = Some(private::exclusive(true));
    match move x {
        Some(ref z) if z.with(|b| *b) => {
            do z.with |b| { assert *b; }
        },
        _ => fail
    }
}