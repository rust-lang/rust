// build-pass

fn main() {
    let _ = &[(); usize::MAX];
}
