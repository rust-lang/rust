// compile-flags: -D while-true
fn main() {
  let mut i = 0;
  while true  { //~ ERROR denote infinite loops with `loop
    i += 1;
    if i == 5 { break; }
  }
}
