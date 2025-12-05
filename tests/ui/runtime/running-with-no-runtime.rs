//@ run-pass
//@ needs-subprocess

#![no_main]

use std::ffi::CStr;
use std::process::{Command, Output};
use std::panic;
use std::str;

#[no_mangle]
extern "C" fn main(argc: core::ffi::c_int, argv: *const *const u8) -> core::ffi::c_int {
    if argc > 1 {
        unsafe {
            match **argv.offset(1) as char {
                '1' => {}
                '2' => println!("foo"),
                '3' => assert!(panic::catch_unwind(|| {}).is_ok()),
                '4' => assert!(panic::catch_unwind(|| panic!()).is_err()),
                '5' => assert!(Command::new("test").spawn().is_err()),
                _ => panic!()
            }
        }
        return 0
    }

    let args = unsafe {
        (0..argc as usize).map(|i| {
            let ptr = *argv.add(i) as *const _;
            CStr::from_ptr(ptr).to_bytes().to_vec()
        }).collect::<Vec<_>>()
    };
    let me = String::from_utf8(args[0].to_vec()).unwrap();

    pass(Command::new(&me).arg("1").output().unwrap());
    pass(Command::new(&me).arg("2").output().unwrap());
    pass(Command::new(&me).arg("3").output().unwrap());
    pass(Command::new(&me).arg("4").output().unwrap());
    pass(Command::new(&me).arg("5").output().unwrap());

    0
}

fn pass(output: Output) {
    if !output.status.success() {
        println!("{:?}", str::from_utf8(&output.stdout));
        println!("{:?}", str::from_utf8(&output.stderr));
    }
}
