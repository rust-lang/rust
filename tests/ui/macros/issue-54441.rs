macro_rules! m {
    () => {
        loop //~ ERROR macro expansion ignores keyword `loop` and any tokens following
    };
}

extern "C" {
    m!();
}

fn main() {}
