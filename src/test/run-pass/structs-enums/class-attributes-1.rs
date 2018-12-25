// run-pass
#![allow(unused_attributes)]
#![allow(non_camel_case_types)]

// pp-exact - Make sure we actually print the attributes
#![feature(custom_attribute)]

struct cat {
    name: String,
}

impl Drop for cat {
    #[cat_dropper]
    fn drop(&mut self) { println!("{} landed on hir feet" , self . name); }
}


#[cat_maker]
fn cat(name: String) -> cat { cat{name: name,} }

pub fn main() { let _kitty = cat("Spotty".to_string()); }
