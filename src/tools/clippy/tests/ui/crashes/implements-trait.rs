//@ check-pass

#[allow(clippy::needless_borrowed_reference)]
fn main() {
    let mut v = Vec::<String>::new();
    let _ = v.iter_mut().filter(|&ref a| a.is_empty());
}
