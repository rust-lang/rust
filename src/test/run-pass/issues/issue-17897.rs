// run-pass
fn action(mut cb: Box<FnMut(usize) -> usize>) -> usize {
    cb(1)
}

pub fn main() {
    println!("num: {}", action(Box::new(move |u| u)));
}
