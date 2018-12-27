// run-pass
#![feature(box_patterns)]

const VALUE: usize = 21;

pub fn main() {
    match &18 {
        &(18..=18) => {}
        _ => { unreachable!(); }
    }
    match &21 {
        &(VALUE..=VALUE) => {}
        _ => { unreachable!(); }
    }
    match Box::new(18) {
        box (18..=18) => {}
        _ => { unreachable!(); }
    }
    match Box::new(21) {
        box (VALUE..=VALUE) => {}
        _ => { unreachable!(); }
    }
}
