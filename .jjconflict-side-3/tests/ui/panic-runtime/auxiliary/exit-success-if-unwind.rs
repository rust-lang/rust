//@ no-prefer-dynamic

#![crate_type = "rlib"]

struct Bomb;

impl Drop for Bomb {
    fn drop(&mut self) {
        std::process::exit(0);
    }
}

pub fn bar(f: fn()) {
    let _bomb = Bomb;
    f();
}
