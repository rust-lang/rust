#![allow(unused_variables)]
#![warn(clippy::vec_init_then_push)]

fn main() {
    let mut def_err: Vec<u32> = Default::default();
    def_err.push(0);

    let mut new_err = Vec::<u32>::new();
    new_err.push(1);

    let mut cap_err = Vec::with_capacity(2);
    cap_err.push(0);
    cap_err.push(1);
    cap_err.push(2);

    let mut cap_ok = Vec::with_capacity(10);
    cap_ok.push(0);

    new_err = Vec::new();
    new_err.push(0);
}
