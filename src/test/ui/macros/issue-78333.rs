// build-pass

#![no_implicit_prelude]

fn main() {
    ::std::panic!();
    ::std::todo!();
    ::std::unimplemented!();
    ::std::assert_eq!(0, 0);
    ::std::dbg!(123);
}
