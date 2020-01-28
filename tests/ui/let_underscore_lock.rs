#![warn(clippy::let_underscore_lock)]

fn main() {
    let m = std::sync::Mutex::new(());
    let rw = std::sync::RwLock::new(());

    let _ = m.lock();
    let _ = rw.read();
    let _ = rw.write();
}
