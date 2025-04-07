//@ check-pass
#![feature(generic_arg_infer)]

fn main() {
    let a: [_; 1] = [String::new(); _];
}
