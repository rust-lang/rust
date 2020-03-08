macro_rules! m {
    () => {
        let //~ ERROR macro expansion ignores token `let` and any following
    };
}

extern "C" {
    m!();
}

fn main() {}
