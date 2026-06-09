// https://github.com/rust-lang/rust/issues/68010
//@ build-pass

fn main() {
    println!("{}", [(); usize::MAX].len());
}
