#![feature(os_string_truncate)]
#![warn(clippy::manual_clear)]

use std::collections::VecDeque;
use std::ffi::OsString;

struct CustomTruncate(String);

impl CustomTruncate {
    fn truncate(&mut self, len: usize) {
        self.0.truncate(len);
    }

    fn clear(&mut self) {
        self.0.clear();
    }
}

fn main() {
    let mut v = vec![1, 2, 3];
    v.truncate(0); //~ manual_clear

    let mut d: VecDeque<i32> = VecDeque::from([1, 2, 3]);
    d.truncate(0); //~ manual_clear

    // lint: macro receiver
    macro_rules! get_vec {
        ($e:expr) => {
            $e
        };
    }
    get_vec!(v).truncate(0); //~ manual_clear

    // no lint: other args
    v.truncate(1);

    // no lint: `0` from a different context
    {
        // `0` inside a block expression should not be changed into `clear()`
        v.truncate({ 0 });
    }

    // lint: String
    let mut s = String::from("abc");
    s.truncate(0); //~ manual_clear

    // lint: OsString
    let mut os = OsString::from("abc");
    os.truncate(0); //~ manual_clear

    // no lint: custom type
    let mut c = CustomTruncate(String::from("abc"));
    c.truncate(0);
}
