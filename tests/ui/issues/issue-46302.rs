fn foo() {
  let s = "abc";
  let u: &str = if true { s[..2] } else { s };
  //~^ ERROR mismatched types
}

fn main() {
    foo();
}
