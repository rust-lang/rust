fn main () {
  let mut line = "";
  let mut i = 0;
  while line != "exit" {
    line = if i == 9 { "exit" } else { "notexit" };
    i += 1;
  }
}
