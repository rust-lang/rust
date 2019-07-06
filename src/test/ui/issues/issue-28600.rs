// build-pass (FIXME(62277): could be check-pass?)
// #28600 ICE: pub extern fn with parameter type &str inside struct impl

struct Test;

impl Test {
    #[allow(dead_code)]
    #[allow(unused_variables)]
    pub extern fn test(val: &str) {

    }
}

fn main() {}
