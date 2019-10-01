macro_rules! m {
    () => {
        let //~ ERROR expected
    };
}

extern "C" {
    m!();
}

fn main() {}
