fn forever2() -> i32 {
  let x: i32 = loop { break }; //~ ERROR mismatched types
  x
}

fn main() {}
