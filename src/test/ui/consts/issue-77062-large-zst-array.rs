// check-pass

fn main() {
    let _ = &[(); usize::MAX];
}
