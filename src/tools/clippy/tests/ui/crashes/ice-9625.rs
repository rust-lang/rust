//@ check-pass

fn main() {
    let x = &1;
    let _ = &1 < x && x < &10;
}
