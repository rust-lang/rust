//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
enum Two { A, B}
impl Drop for Two {
    fn drop(&mut self) {
        println!("Dropping!");
    }
}
fn main() {
    let k = Two::A;
}
