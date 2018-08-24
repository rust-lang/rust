#[derive(Debug)]
struct r {
  b: bool,
}

impl Drop for r {
    fn drop(&mut self) {}
}

fn main() {
    let i = Box::new(r { b: true });
    let _j = i.clone(); //~ ERROR no method named `clone` found
    println!("{:?}", i);
}
