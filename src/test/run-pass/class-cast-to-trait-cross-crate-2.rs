// xfail-fast
// aux-build:cci_class_cast.rs
use cci_class_cast;
import to_str::ToStr;
import cci_class_cast::kitty::*;

fn print_out<T: ToStr>(thing: T, expected: ~str) {
  let actual = thing.to_str();
  debug!("%s", actual);
  assert(actual == expected);
}

fn main() {
  let nyan : ToStr  = cat(0u, 2, ~"nyan") as ToStr;
  print_out(nyan, ~"nyan");
}

