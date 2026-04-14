#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use dirs::home_dir;
use std::env::current_dir;
use std::env::set_current_dir;
use std::path::PathBuf;

pub fn change_directory(dir: Option<&str>) -> Option<PathBuf> {
    let mut current: PathBuf = current_dir().unwrap();

    match dir {
        Some(dir) if dir == ".." => {
            current.pop();
            match set_current_dir(&current) {
                Ok(_) => Some(current),
                _ => return None,
            }
        }
        Some(dir) if dir == "." => {
            let buff = PathBuf::from(dir);
            Some(buff)
        }
        Some(dir) => {
            current.push(dir);
            match set_current_dir(&current) {
                Ok(_) => Some(current),
                _ => return None,
            }
        }
        None => match set_current_dir(home_dir().unwrap()) {
            Ok(_) => Some(home_dir().unwrap()),
            _ => return None,
        },
    }
}
