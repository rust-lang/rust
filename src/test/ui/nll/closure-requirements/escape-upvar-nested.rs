// As in `escape-upvar-ref.rs`, test closure that:
//
// - captures a variable `y`
// - stores reference to `y` into another, longer-lived spot
//
// except that the closure does so via a second closure.

// compile-flags:-Zborrowck=mir -Zverbose

#![feature(rustc_attrs)]

#[rustc_regions]
fn test() {
    let x = 44;
    let mut p = &x;

    {
        let y = 22;

        let mut closure = || {
            let mut closure1 = || p = &y; //~ ERROR `y` does not live long enough [E0597]
            closure1();
        };

        closure();
    }

    deref(p);
}

fn deref(_p: &i32) { }

fn main() { }
