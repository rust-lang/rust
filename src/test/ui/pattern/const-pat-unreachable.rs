#![feature(const_transmute)]
#![deny(unreachable_patterns)]

const FOO: &[u8] = unsafe { std::mem::transmute::<usize, &[u8; 0]>(1) };
const BAR: &[u8] = unsafe { std::mem::transmute::<usize, &[u8; 0]>(3) };

fn main() {
    let x: &[u8] = unimplemented!();
    match x {
        b"" => {}
        FOO => {} //~ ERROR unreachable pattern
        BAR => {} //~ ERROR unreachable pattern
        _ => {}
    }
}
