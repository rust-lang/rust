#![allow(stable_features)]
// ignore-cloudabi no processes
// ignore-emscripten no processes
// ignore-sgx no processes

#![feature(process_try_wait)]

use std::env;
use std::process::Command;
use std::thread;
use std::time::Duration;

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 1 {
        match &args[1][..] {
            "sleep" => thread::sleep(Duration::new(1_000, 0)),
            _ => {}
        }
        return
    }

    let mut me = Command::new(env::current_exe().unwrap())
                         .arg("sleep")
                         .spawn()
                         .unwrap();
    let maybe_status = me.try_wait().unwrap();
    assert!(maybe_status.is_none());
    let maybe_status = me.try_wait().unwrap();
    assert!(maybe_status.is_none());

    me.kill().unwrap();
    me.wait().unwrap();

    let status = me.try_wait().unwrap().unwrap();
    assert!(!status.success());
    let status = me.try_wait().unwrap().unwrap();
    assert!(!status.success());

    let mut me = Command::new(env::current_exe().unwrap())
                         .arg("return-quickly")
                         .spawn()
                         .unwrap();
    loop {
        match me.try_wait() {
            Ok(Some(res)) => {
                assert!(res.success());
                break
            }
            Ok(None) => {
                thread::sleep(Duration::from_millis(1));
            }
            Err(e) => panic!("error in try_wait: {}", e),
        }
    }

    let status = me.try_wait().unwrap().unwrap();
    assert!(status.success());
}
