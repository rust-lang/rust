fn main() {
    println!("Custom backtrace: {}", std::backtrace::Backtrace::forced_capture());
    //~^ ERROR no function or associated item name
}
