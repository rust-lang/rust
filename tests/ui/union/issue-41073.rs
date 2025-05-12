union Test {
    a: A, //~ ERROR field must implement `Copy` or be wrapped in `ManuallyDrop<...>` to be used in a union
    b: B
}

#[derive(Debug)]
struct A(i32);
impl Drop for A {
    fn drop(&mut self) { println!("A"); }
}

#[derive(Debug)]
struct B(f32);
impl Drop for B {
    fn drop(&mut self) { println!("B"); }
}

fn main() {
    let mut test = Test { a: A(3) };
    println!("{:?}", unsafe { test.b });
    unsafe { test.b = B(0.5); }
}
