//@ edition: 2024
//@ compile-flags: -Z validate-mir

#![deny(if_let_rescope)]

struct Droppy;
impl Drop for Droppy {
    fn drop(&mut self) {
        println!("dropped");
    }
}
impl Droppy {
    fn get_ref(&self) -> Option<&u8> {
        None
    }
}

fn do_something<T>(_: &T) {}

fn main() {
    do_something(if let Some(value) = Droppy.get_ref() { value } else { &0 });
    //~^ ERROR: temporary value dropped while borrowed
    do_something(if let Some(value) = Droppy.get_ref() {
        //~^ ERROR: temporary value dropped while borrowed
        value
    } else if let Some(value) = Droppy.get_ref() {
        //~^ ERROR: temporary value dropped while borrowed
        value
    } else {
        &0
    });
}
