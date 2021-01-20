extern crate a;

#[link(name = "b", kind = "static")]
extern {
    pub fn b();
}


fn main() {
    unsafe { b(); }
}
