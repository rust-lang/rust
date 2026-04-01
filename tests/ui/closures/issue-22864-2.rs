//@ run-pass
//@ needs-threads

pub fn main() {
    let f = || || 0;
    std::thread::spawn(f());
}
