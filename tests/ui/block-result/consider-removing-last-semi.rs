//@ run-rustfix

pub fn f() -> String {  //~ ERROR mismatched types
    0u8;
    "bla".to_string();
}

pub fn g() -> String {  //~ ERROR mismatched types
    "this won't work".to_string();
    "removeme".to_string();
}

pub fn macro_tests() -> u32 {  //~ ERROR mismatched types
    macro_rules! mac {
        () => (1);
    }
    mac!();
}

fn main() {}
