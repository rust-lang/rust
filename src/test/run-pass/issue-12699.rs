// ignore-wasm32-bare can't block the thread

use std::thread;

fn main() {
    thread::sleep_ms(250);
}
