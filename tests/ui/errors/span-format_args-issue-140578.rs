fn check_format_args() {
  print!("{:?} {a} {a:?}", [], a = 1 + 1);
  //~^ ERROR type annotations needed
}

fn check_format_args_nl() {
  println!("{:?} {a} {a:?}", [], a = 1 + 1);
  //~^ ERROR type annotations needed
}

fn main() {}
