fn main() {
    println!("Custom backtrace: {}", std::backtrace::Backtrace::forced_capture());
    //~^ ERROR no associated function or constant name
}
