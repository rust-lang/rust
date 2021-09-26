// edition:2021
// run-pass
#![feature(if_let_guard)]
#[allow(unused_must_use)]
#[allow(dead_code)]

fn print_error_count(registry: &Registry) {
    |x: &Registry| {
        match &x {
            Registry if let _ = registry.try_find_description() => { }
            //~^ WARNING: irrefutable `if let` guard pattern
            _ => {}
        }
    };
}

struct Registry;
impl Registry {
    pub fn try_find_description(&self) {
        unimplemented!()
    }
}

fn main() {}
