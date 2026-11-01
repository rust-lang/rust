fn check_format_args() {
  print!("{:?} {a} {a:?}", [], a = 1 + 1);
  //~^ ERROR type annotations needed
}

fn check_format_args_nl() {
  println!("{:?} {a} {a:?}", [], a = 1 + 1);
  //~^ ERROR type annotations needed
}

fn check_multi1() {
  println!("{:?} {:?} {a} {a:?}", [], [], a = 1 + 1);
  //~^ ERROR type annotations needed
}

fn check_multi2() {
  println!("{:?} {:?} {a} {a:?} {b:?}", [], [], a = 1 + 1, b = []);
  //~^ ERROR type annotations needed
}

fn check_unformatted() {
  println!("
  {:?} {:?}
{a}
{a:?}",
        [],
        //~^ ERROR type annotations needed
 [],
a = 1 + 1);
}

fn main() {}
