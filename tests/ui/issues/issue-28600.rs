//@ build-pass
// #28600 ICE: pub extern fn with parameter type &str inside struct impl

struct Test;

impl Test {
    #[allow(dead_code)]
    #[allow(unused_variables)]
    #[allow(improper_c_fn_definitions)]
    pub extern "C" fn test(val: &str) {

    }
}

fn main() {}
