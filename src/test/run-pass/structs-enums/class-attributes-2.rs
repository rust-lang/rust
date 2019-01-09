// run-pass
#![allow(unused_attributes)]
#![allow(non_camel_case_types)]

#![feature(custom_attribute)]

struct cat {
  name: String,
}

impl Drop for cat {
    #[cat_dropper]
    /**
       Actually, cats don't always land on their feet when you drop them.
    */
    fn drop(&mut self) {
        println!("{} landed on hir feet", self.name);
    }
}

#[cat_maker]
/**
Maybe it should technically be a kitten_maker.
*/
fn cat(name: String) -> cat {
    cat {
        name: name
    }
}

pub fn main() {
  let _kitty = cat("Spotty".to_string());
}
