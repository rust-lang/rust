// ignore-windows: We do not check leaks on Windows

//error-pattern: the evaluated program leaked memory

fn main() {
    std::mem::forget(Box::new(42));
}
