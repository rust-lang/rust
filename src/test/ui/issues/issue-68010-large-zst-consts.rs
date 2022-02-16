// check-pass

fn main() {
    println!("{}", [(); usize::MAX].len());
}
