macro_rules! m {
    //~^ ERROR missing `fn`, `type`, `const`, or `static` for item declaration
    () => {
        let
    };
}

extern "C" {
    m!();
}

fn main() {}
