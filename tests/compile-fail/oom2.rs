#![feature(custom_attribute, attr_literals)]
#![miri(memory_size=1000)]

fn bar(i: i32) {
    if i < 1000 { //~ERROR tried to allocate 4 more bytes, but only 0 bytes are free of the 1000 byte memory
        bar(i + 1)
        //~^NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
        //~|NOTE inside call to bar
    }
}

fn main() { //~NOTE inside call to main
    bar(1);
    //~^NOTE inside call to bar
}
