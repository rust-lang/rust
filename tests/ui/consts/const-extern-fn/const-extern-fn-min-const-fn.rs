const extern "C" fn ptr_cast(val: *const u8) {
    val as usize;
    //~^ ERROR pointers cannot be cast to integers
}

fn main() {}
