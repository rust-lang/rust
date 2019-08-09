fn warn(_: &str) {}

macro_rules! intrinsic_match {
    ($intrinsic:expr) => {
        warn(format!("unsupported intrinsic {}", $intrinsic));
        //~^ ERROR mismatched types
    };
}

fn main() {
    intrinsic_match! {
        "abc"
    };
}
