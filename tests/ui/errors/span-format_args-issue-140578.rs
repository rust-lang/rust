fn main() {
  println!("{:?} {a} {a:?}", [], a = 1 + 1);
  //~^ ERROR type annotations needed
}
