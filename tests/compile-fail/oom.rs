#![feature(custom_attribute, attr_literals)]
#![miri(memory_size=20)]

fn main() {
    let _x = [42; 10];
    //~^ERROR tried to allocate 40 more bytes, but only
}
