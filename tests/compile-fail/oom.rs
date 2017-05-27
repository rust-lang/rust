#![feature(custom_attribute, attr_literals)]
#![miri(memory_size=4095)]

fn main() {
    let _x = [42; 1024];
    //~^ERROR tried to allocate 4096 more bytes, but only
}
