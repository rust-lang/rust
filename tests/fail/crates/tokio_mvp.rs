//@compile-flags: -Zmiri-disable-isolation
//@error-pattern: can't call foreign function: epoll_create1
//@normalize-stderr-test: "   = note: inside .*\n" -> ""
//@only-target-linux: the errors differ too much between platforms

#[tokio::main]
async fn main() {}
