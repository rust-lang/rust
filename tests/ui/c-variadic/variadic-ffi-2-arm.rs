//@ only-arm
//@ build-pass
#![feature(extended_varargs_abi_support)]

fn aapcs(f: extern "aapcs" fn(usize, ...)) {
    f(22, 44);
}

fn main() {}
