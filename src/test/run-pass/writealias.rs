#![allow(dead_code)]

use std::sync::Mutex;

struct Point {x: isize, y: isize, z: isize}

fn f(p: &mut Point) { p.z = 13; }

pub fn main() {
    let x = Some(Mutex::new(true));
    match x {
        Some(ref z) if *z.lock().unwrap() => {
            assert!(*z.lock().unwrap());
        },
        _ => panic!()
    }
}
