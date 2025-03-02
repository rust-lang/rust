#![warn(clippy::readonly_write_lock)]

use std::sync::RwLock;

fn mutate_i32(x: &mut i32) {
    *x += 1;
}

fn accept_i32(_: i32) {}

fn main() {
    let lock = RwLock::new(42);
    let lock2 = RwLock::new(1234);

    {
        let writer = lock.write().unwrap();
        //~^ readonly_write_lock

        dbg!(&writer);
    }

    {
        let writer = lock.write().unwrap();
        //~^ readonly_write_lock

        accept_i32(*writer);
    }

    {
        let mut writer = lock.write().unwrap();
        mutate_i32(&mut writer);
        dbg!(&writer);
    }

    {
        let mut writer = lock.write().unwrap();
        *writer += 1;
    }

    {
        let mut writer1 = lock.write().unwrap();
        let mut writer2 = lock2.write().unwrap();
        *writer2 += 1;
        *writer1 = *writer2;
    }
}

fn issue12733(rw: &RwLock<()>) {
    let _write_guard = rw.write().unwrap();
}
