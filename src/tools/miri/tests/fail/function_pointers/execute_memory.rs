// Validation makes this fail in the wrong place
//@compile-flags: -Zmiri-disable-validation

fn main() {
    let x = Box::new(42);
    unsafe {
        let f = std::mem::transmute::<Box<i32>, fn()>(x);
        f() //~ ERROR: function pointer but it does not point to a function
    }
}
