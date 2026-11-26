extern crate aux;

#[no_mangle]
#[inline(never)]
pub fn caller(x: u64) -> u64 {
    aux::aux_callee(x)
}

#[no_mangle]
#[inline(never)]
pub fn wrapper(x: u64) -> u64 {
    caller(x)
}

fn main() {
    println!("{}", wrapper(3));
}
