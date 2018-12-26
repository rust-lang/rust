#[derive(Debug)]
struct R {
  i:isize
}

fn r(i:isize) -> R { R { i: i } }

impl Drop for R {
    fn drop(&mut self) {}
}

fn main() {
    // This can't make sense as it would copy the classes
    let i = vec![r(0)];
    let j = vec![r(1)];
    let k = i + j;
    //~^ ERROR binary operation `+` cannot be applied to type
    println!("{:?}", j);
}
