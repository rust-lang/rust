//@ check-pass
fn useful(i: usize) -> usize {
    i
}

fn useful2(i: usize) -> usize {
    i
}

fn main() {
    for f in &[useful, useful2, |x| x] {
        println!("{}", f(6));
    }
}
