macro_rules! m {
    //~^ ERROR missing `fn`, `type`, or `static` for extern-item declaration
    () => {
        let
    };
}

extern "C" {
    m!();
}

fn main() {}
