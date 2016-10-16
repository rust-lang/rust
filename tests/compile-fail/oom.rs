#![feature(custom_attribute, attr_literals)]
#![miri(memory_size=0)]

fn main() {
    let _x = [42; 10];
    //~^ERROR tried to allocate 40 more bytes, but only 0 bytes are free of the 0 byte memory
}
