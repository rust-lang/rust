#![allow(deref_nullptr)]

enum Never {}

fn main() {
    unsafe {
        match *std::ptr::null::<Result<Never, Never>>() {
        //~^ ERROR: read discriminant of an uninhabited enum variant
            Ok(_) => {
                lol();
            }
            Err(_) => {
                wut();
            }
        }
    }
}

fn lol() {
    println!("lol");
}

fn wut() {
    println!("wut");
}
