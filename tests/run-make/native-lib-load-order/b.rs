extern crate a;

#[link(name = "b", kind = "static")]
extern "C" {
    pub fn b();
}

fn main() {
    unsafe {
        b();
    }
}
