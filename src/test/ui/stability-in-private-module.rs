// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl

fn main() {
    let _ = std::thread::thread_info::current_thread();
    //~^ERROR module `thread_info` is private
}
