// error-pattern:casting

struct A;

fn main() {
  println!("{:?}", 1.0 as *const A); // Can't cast float to foreign.
}
