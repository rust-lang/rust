//@ check-pass

fn main() {
    let _unused = if true {
        core::ptr::copy::<i32>
    } else {
        core::ptr::copy_nonoverlapping::<i32>
    };
}
