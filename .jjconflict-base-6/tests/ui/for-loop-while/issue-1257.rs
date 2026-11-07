//@ run-pass

pub fn main () {
  let mut line = "".to_string();
  let mut i = 0;
  while line != "exit".to_string() {
    line = if i == 9 { "exit".to_string() } else { "notexit".to_string() };
    i += 1;
  }
}
