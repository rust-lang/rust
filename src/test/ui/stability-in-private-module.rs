fn main() {
    let _ = std::thread::thread_info::current_thread();
    //~^ERROR module `thread_info` is private
}
