// revisions: mir thir
// [thir]compile-flags: -Zthir-unsafeck

#[deny(unused_unsafe)]
fn main() {
    let mut v = Vec::<i32>::with_capacity(24);

    unsafe {
        let f = |v: &mut Vec<_>| {
            unsafe { //~ ERROR unnecessary `unsafe`
                v.set_len(24);
                |w: &mut Vec<u32>| { unsafe { //~ ERROR unnecessary `unsafe`
                    w.set_len(32);
                } };
            }
            |x: &mut Vec<u32>| { unsafe { //~ ERROR unnecessary `unsafe`
                x.set_len(40);
            } };
        };

        v.set_len(0);
        f(&mut v);
    }

    |y: &mut Vec<u32>| { unsafe {
        y.set_len(48);
    } };
}
