// run-pass
// ignore-emscripten no threads support

pub fn main() {
    let f = || || 0;
    std::thread::spawn(f());
}
