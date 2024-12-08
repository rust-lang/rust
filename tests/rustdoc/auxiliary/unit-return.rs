pub fn f2<F: FnMut(u32) + Clone>(f: F) {}

pub fn f3<F: FnMut(u64) -> () + Clone>(f: F) {}
