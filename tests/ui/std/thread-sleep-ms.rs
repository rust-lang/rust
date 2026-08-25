//@ run-pass
//@ needs-threads
//@ ignore-emscripten (FIXME: test hangs on emscripten)

#![allow(deprecated)]

fn main() {
    std::thread::sleep_ms(250);
}
