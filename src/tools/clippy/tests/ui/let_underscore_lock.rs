#![warn(clippy::let_underscore_lock)]

extern crate parking_lot;

fn main() {
    let m = std::sync::Mutex::new(());
    let rw = std::sync::RwLock::new(());

    let _ = m.lock();
    let _ = rw.read();
    let _ = rw.write();
    let _ = m.try_lock();
    let _ = rw.try_read();
    let _ = rw.try_write();

    use parking_lot::{lock_api::RawMutex, Mutex, RwLock};

    let p_m: Mutex<()> = Mutex::const_new(RawMutex::INIT, ());
    let _ = p_m.lock();

    let p_m1 = Mutex::new(0);
    let _ = p_m1.lock();

    let p_rw = RwLock::new(0);
    let _ = p_rw.read();
    let _ = p_rw.write();
}
