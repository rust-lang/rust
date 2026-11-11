macro_rules! please_recover {
    ($a:expr) => {};
}

please_recover! { not 1 }
//~^ ERROR unexpected `1` after identifier

fn main() {}
