//@ only-arm
//@ build-pass

fn aapcs(f: extern "aapcs" fn(usize, ...)) {
    f(22, 44);
}

fn main() {}
