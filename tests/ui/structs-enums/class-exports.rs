//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

/* Test that exporting a class also exports its
   public fields and methods */

use kitty::cat;

mod kitty {
    pub struct cat {
        meows: usize,
        name: String,
    }

    impl cat {
        pub fn get_name(&self) -> String { self.name.clone() }
    }

    pub fn cat(in_name: String) -> cat {
        cat {
            name: in_name,
            meows: 0
        }
    }
}

pub fn main() {
  assert_eq!(cat("Spreckles".to_string()).get_name(),
                 "Spreckles".to_string());
}
