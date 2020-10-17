// build-pass

fn main() {
    println!("{}", [(); std::usize::MAX].len());
}
