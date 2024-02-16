//@ ignore-sgx std::os::fortanix_sgx::usercalls::raw::Result changes compiler suggestions

pub struct GslResult {
    pub val: f64,
    pub err: f64
}

impl GslResult {
    pub fn new() -> GslResult {
        Result {
//~^ ERROR expected struct, variant or union type, found enum `Result`
            val: 0f64,
            err: 0f64
        }
    }
}

fn main() {}
