// aux-build:cci_class_6.rs
use cci_class_6;
import cci_class_6::kitties::*;

fn main() {
  let nyan : cat<char> = cat::<char>(52u, 99, ~['p']);
  let kitty = cat(1000u, 2, ~["tabby"]);
  assert(nyan.how_hungry == 99);
  assert(kitty.how_hungry == 2);
  nyan.speak(~[1u,2u,3u]);
  assert(nyan.meow_count() == 55u);
  kitty.speak(~["meow", "mew", "purr", "chirp"]);
  assert(kitty.meow_count() == 1004u);
}
