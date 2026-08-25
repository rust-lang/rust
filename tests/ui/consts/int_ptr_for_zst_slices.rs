//@ check-pass

const FOO: &str = unsafe { &*(1_usize as *const [u8; 0] as *const [u8] as *const str) };

fn main() {}
