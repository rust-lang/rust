//@ check-pass

fn main() {
    const PAT: u8 = 1;

    match 0 {
        (.. PAT) => {}
        _ => {}
    }
}
