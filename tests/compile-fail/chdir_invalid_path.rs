// compile-flags: -Zmiri-disable-isolation

extern {
    pub fn chdir(dir: *const u8) -> i32;
}

fn main() {
    let path = vec![0xc3u8, 0x28, 0];
    // test that `chdir` errors with invalid utf-8 path
    unsafe { chdir(path.as_ptr()) };  //~ ERROR is not a valid utf-8 string
}
