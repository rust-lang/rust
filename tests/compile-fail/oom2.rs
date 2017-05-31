#![feature(box_syntax, custom_attribute, attr_literals)]
#![miri(memory_size=2048)]

fn main() {
    loop {
        ::std::mem::forget(box 42); //~ ERROR tried to allocate 4 more bytes
    }
}
