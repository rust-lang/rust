pub struct Option;
impl Option {
    pub fn unwrap(self) {}
}

/// [`Option::unwrap`]
pub mod task {}

extern "C" {
    pub fn main() -> std::os::raw::c_int;
}
