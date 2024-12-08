struct A;

fn main() {
  println!("{:?}", 1.0 as *const A); //~ERROR  casting `f64` as `*const A` is invalid
}
