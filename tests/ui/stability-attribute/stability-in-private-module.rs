fn main() {
    let _ = std::sys::os::errno();
    //~^ERROR module `sys` is private
}
