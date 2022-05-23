// compile-flags: -Zmiri-check-number-validity

fn main() {
    let r = &mut 42;
    let _i: usize = unsafe { std::mem::transmute(r) }; //~ ERROR expected plain (non-pointer) bytes
}
