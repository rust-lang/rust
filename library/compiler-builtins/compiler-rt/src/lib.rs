#![feature(drop_types_in_const)]

extern crate libloading;

use std::sync::{Once, ONCE_INIT};
use std::env;

use libloading::Library;

fn compiler_rt() -> &'static Library {
    let dir = env::current_exe().unwrap();
    let cdylib = dir.parent().unwrap().read_dir().unwrap().map(|c| {
        c.unwrap().path()
    }).find(|path| {
        path.file_name().unwrap().to_str().unwrap().contains("compiler_rt_cdylib")
    }).unwrap();

    unsafe {
        static mut COMPILER_RT: Option<Library> = None;
        static INIT: Once = ONCE_INIT;

        INIT.call_once(|| {
            COMPILER_RT = Some(Library::new(&cdylib).unwrap());
        });
        COMPILER_RT.as_ref().unwrap()
    }
}

pub fn get(sym: &str) -> usize {
    unsafe {
        let sym = format!("_{}", sym);
        let f: fn() -> usize = *compiler_rt().get(sym.as_bytes()).unwrap();
        f()
    }
}
