// check-pass
// compile-flags: -Z unpretty=expanded

#![feature(cfg_match)]

trait Trait {
    fn impl_fn(&self);
}
struct Struct;
impl Trait for Struct {
    cfg_match! {
        cfg(feature = "blah") => { fn impl_fn(&self) {} }
        _ => { fn impl_fn(&self) {} }
    }
}

cfg_match! {
    cfg(unix) => { fn item() {} }
    _ => { fn item() {} }
}

fn statement() {
    cfg_match! {
        cfg(unix) => { fn statement() {} }
        _ => { fn statement() {} }
    }
}

pub fn main() {}
