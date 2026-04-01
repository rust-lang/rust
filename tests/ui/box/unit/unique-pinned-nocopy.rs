#[derive(Debug)]
struct R {
  b: bool,
}

impl Drop for R {
    fn drop(&mut self) {}
}

fn main() {
    let i = Box::new(R { b: true });
    let _j = i.clone(); //~ ERROR the method
    println!("{:?}", i);
}
