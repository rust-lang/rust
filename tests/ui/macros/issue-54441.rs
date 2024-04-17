macro_rules! m {
    () => {
        let //~ ERROR macro expansion ignores keyword `let` and any tokens following
    };
}

extern "C" {
    m!();
}

fn main() {}
