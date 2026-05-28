//@check-pass
//@edition: 2018

// https://github.com/rust-lang/rust/issues/98291

macro_rules! wrap {
    () => {
        macro_rules! _a {
            () => {
                "auxiliary/macro-include-items-expr.rs"
            };
        }
        macro_rules! _env_name {
            () => {
                "PATH"
            }
        }
    };
}

wrap!();

use _a as a;
use _env_name as env_name;

fn main() {
    format_args!(a!());
    include!(a!());
    include_str!(a!());
    include_bytes!(a!());
    env!(env_name!());
    option_env!(env_name!());
}
