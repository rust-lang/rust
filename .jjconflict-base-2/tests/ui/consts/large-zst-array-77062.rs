//@ build-pass
pub static FOO: [(); usize::MAX] = [(); usize::MAX];

fn main() {
    let _ = &[(); usize::MAX];
}
