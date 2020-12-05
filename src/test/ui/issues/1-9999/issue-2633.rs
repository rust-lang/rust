// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

#[derive(Copy, Clone)]
struct cat {
    meow: extern "Rust" fn(),
}

fn meow() {
    println!("meow")
}

fn cat() -> cat {
    cat {
        meow: meow,
    }
}

#[derive(Copy, Clone)]
struct KittyInfo {kitty: cat}

// Code compiles and runs successfully if we add a + before the first arg
fn nyan(kitty: cat, _kitty_info: KittyInfo) {
    (kitty.meow)();
}

pub fn main() {
    let kitty = cat();
    nyan(kitty, KittyInfo {kitty: kitty});
}
