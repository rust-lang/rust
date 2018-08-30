// ignore-windows: We do not check leaks on Windows
// ignore-macos: We do not check leaks on macOS

//error-pattern: the evaluated program leaked memory

fn main() {
    std::mem::forget(Box::new(42));
}
