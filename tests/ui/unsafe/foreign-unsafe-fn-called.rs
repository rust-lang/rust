mod test {
    extern "C" {
        pub fn free();
    }
}

fn main() {
    test::free();
    //~^ ERROR call to unsafe function `test::free` is unsafe
}
