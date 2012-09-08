/* Test that exporting a class also exports its
   public fields and methods */

use kitty::*;

mod kitty {
  export cat;
  struct cat {
    meows: uint,
    name: ~str,
  }

  impl cat {
    fn get_name() -> ~str {  self.name }
  }

    fn cat(in_name: ~str) -> cat {
        cat {
            name: in_name,
            meows: 0u
        }
    }

}

fn main() {
  assert(cat(~"Spreckles").get_name() == ~"Spreckles");
}