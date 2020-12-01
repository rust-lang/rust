// build-pass

#![no_implicit_prelude]

fn main() {
    ::std::panic!();
    ::std::todo!();
    ::std::unimplemented!();
    ::std::assert_eq!(0, 0);
    ::std::assert_ne!(0, 1);
    ::std::dbg!(123);
    ::std::unreachable!();
}
