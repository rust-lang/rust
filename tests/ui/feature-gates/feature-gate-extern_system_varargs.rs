fn system(f: extern "system" fn(usize, ...)) {
    //~^  ERROR unstable

    f(22, 44);
}

fn main() {}
