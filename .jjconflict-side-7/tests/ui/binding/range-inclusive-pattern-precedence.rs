//@ run-pass

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
}
