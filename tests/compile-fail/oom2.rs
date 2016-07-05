#![feature(custom_attribute)]
#![miri(memory_size="1000")]

fn bar(i: i32) {
    if i < 1000 {
        bar(i + 1) //~ ERROR tried to allocate 4 more bytes, but only 1 bytes are free of the 1000 byte memory
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
    }
}

fn main() { //~NOTE inside call to main
    bar(1);
    //~^NOTE inside call to bar
}
