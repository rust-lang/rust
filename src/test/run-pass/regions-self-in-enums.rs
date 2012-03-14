enum int_wrapper {
    int_wrapper_ctor(&int)
}

fn main() {
    unsafe {
        let x = 3;
        let y = int_wrapper_ctor(&x);
        let z : &int;
        alt y {
            int_wrapper_ctor(zz) { z = zz; }
        }
        log(debug, *z);
    }
}

