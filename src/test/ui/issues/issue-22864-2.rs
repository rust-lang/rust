// run-pass
// ignore-emscripten no threads support
// ignore-uefi no threads support

pub fn main() {
    let f = || || 0;
    std::thread::spawn(f());
}
