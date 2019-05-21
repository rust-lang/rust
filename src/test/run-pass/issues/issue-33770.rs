// run-pass
// ignore-cloudabi no processes
// ignore-emscripten no processes
// ignore-sgx no processes

use std::process::{Command, Stdio};
use std::env;
use std::sync::{Mutex, RwLock};
use std::time::Duration;
use std::thread;

fn test_mutex() {
    let m = Mutex::new(0);
    let _g = m.lock().unwrap();
    let _g2 = m.lock().unwrap();
}

fn test_try_mutex() {
    let m = Mutex::new(0);
    let _g = m.lock().unwrap();
    let _g2 = m.try_lock().unwrap();
}

fn test_rwlock_ww() {
    let m = RwLock::new(0);
    let _g = m.write().unwrap();
    let _g2 = m.write().unwrap();
}

fn test_try_rwlock_ww() {
    let m = RwLock::new(0);
    let _g = m.write().unwrap();
    let _g2 = m.try_write().unwrap();
}

fn test_rwlock_rw() {
    let m = RwLock::new(0);
    let _g = m.read().unwrap();
    let _g2 = m.write().unwrap();
}

fn test_try_rwlock_rw() {
    let m = RwLock::new(0);
    let _g = m.read().unwrap();
    let _g2 = m.try_write().unwrap();
}

fn test_rwlock_wr() {
    let m = RwLock::new(0);
    let _g = m.write().unwrap();
    let _g2 = m.read().unwrap();
}

fn test_try_rwlock_wr() {
    let m = RwLock::new(0);
    let _g = m.write().unwrap();
    let _g2 = m.try_read().unwrap();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        match &*args[1] {
            "mutex" => test_mutex(),
            "try_mutex" => test_try_mutex(),
            "rwlock_ww" => test_rwlock_ww(),
            "try_rwlock_ww" => test_try_rwlock_ww(),
            "rwlock_rw" => test_rwlock_rw(),
            "try_rwlock_rw" => test_try_rwlock_rw(),
            "rwlock_wr" => test_rwlock_wr(),
            "try_rwlock_wr" => test_try_rwlock_wr(),
            _ => unreachable!(),
        }
        // If we reach this point then the test failed
        println!("TEST FAILED: {}", args[1]);
    } else {
        let mut v = vec![];
        v.push(Command::new(&args[0]).arg("mutex").stderr(Stdio::null()).spawn().unwrap());
        v.push(Command::new(&args[0]).arg("try_mutex").stderr(Stdio::null()).spawn().unwrap());
        v.push(Command::new(&args[0]).arg("rwlock_ww").stderr(Stdio::null()).spawn().unwrap());
        v.push(Command::new(&args[0]).arg("try_rwlock_ww").stderr(Stdio::null()).spawn().unwrap());
        v.push(Command::new(&args[0]).arg("rwlock_rw").stderr(Stdio::null()).spawn().unwrap());
        v.push(Command::new(&args[0]).arg("try_rwlock_rw").stderr(Stdio::null()).spawn().unwrap());
        v.push(Command::new(&args[0]).arg("rwlock_wr").stderr(Stdio::null()).spawn().unwrap());
        v.push(Command::new(&args[0]).arg("try_rwlock_wr").stderr(Stdio::null()).spawn().unwrap());

        thread::sleep(Duration::new(1, 0));

        // Make sure all subprocesses either panicked or were killed because they deadlocked
        for mut c in v {
            c.kill().ok();
            assert!(!c.wait().unwrap().success());
        }
    }
}
