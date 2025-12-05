fn main() {
  let s = "abc";
  let t = if true { s[..2] } else { s };
  //~^ ERROR `if` and `else` have incompatible types
  let u: &str = if true { s[..2] } else { s };
  //~^ ERROR mismatched types
  let v = s[..2];
  //~^ ERROR the size for values of type
  let w: &str = s[..2];
  //~^ ERROR mismatched types
}
