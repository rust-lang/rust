//@ check-pass

fn main() {
    #[clippy::author]
    let _ = ::std::cmp::min(3, 4);
}
