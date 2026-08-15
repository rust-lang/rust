fn main() {
    let _ = std::sys::io::errno();
    //~^ERROR module `sys` is private
}
