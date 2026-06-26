//@ check-pass

#[expect(unused)]
trait UnusedTrait {}

struct UsedStruct(u32);

impl UnusedTrait for UsedStruct {}

fn main() {
    let x = UsedStruct(12);
    println!("Hello World! {}", x.0);
}
