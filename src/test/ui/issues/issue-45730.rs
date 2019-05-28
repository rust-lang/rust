use std::fmt;
fn main() {
    let x: *const _ = 0 as _; //~ ERROR cannot cast

    let x: *const _ = 0 as *const _; //~ ERROR cannot cast
    let y: Option<*const dyn fmt::Debug> = Some(x) as _;

    let x = 0 as *const i32 as *const _ as *mut _; //~ ERROR cannot cast
}
