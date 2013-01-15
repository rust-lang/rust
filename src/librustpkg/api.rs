use core::*;

pub struct Crate {
    file: ~str,
    flags: ~[~str],
    cfg: ~[~str]
}

pub impl Crate {
    fn flag(flag: ~str) -> Crate {
        Crate {
            flags: vec::append(self.flags, flag),
            .. copy self
        }
    }
}

pub fn build(_targets: ~[Crate]) {
    // TODO: magic
}
