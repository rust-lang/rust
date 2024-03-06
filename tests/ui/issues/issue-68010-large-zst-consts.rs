//@ build-pass

fn main() {
    println!("{}", [(); usize::MAX].len());
}
