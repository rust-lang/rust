//@ run-pass
fn main() {
    let x = Box::new([1, 2, 3]);
    let y = x as Box<[i32]>;
    println!("y: {:?}", y);
}
