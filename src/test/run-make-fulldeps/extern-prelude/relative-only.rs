// Extern prelude names are not available by absolute paths

#![feature(extern_prelude)]

use ep_lib::S;

fn main() {
    let s = ::ep_lib::S;
}
