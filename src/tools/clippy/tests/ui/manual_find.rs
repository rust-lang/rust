#![allow(unused)]
#![warn(clippy::manual_find)]
//@no-rustfix
fn vec_string(strings: Vec<String>) -> Option<String> {
    for s in strings {
        //~^ manual_find

        if s == String::new() {
            return Some(s);
        }
    }
    None
}

fn tuple(arr: Vec<(String, i32)>) -> Option<String> {
    for (s, _) in arr {
        //~^ manual_find

        if s == String::new() {
            return Some(s);
        }
    }
    None
}

mod issue9521 {
    fn condition(x: u32, y: u32) -> Result<bool, ()> {
        todo!()
    }

    fn find_with_early_return(v: Vec<u32>) -> Option<u32> {
        for x in v {
            if condition(x, 10).ok()? {
                return Some(x);
            }
        }
        None
    }

    fn find_with_early_break(v: Vec<u32>) -> Option<u32> {
        for x in v {
            if if x < 3 {
                break;
            } else {
                x < 10
            } {
                return Some(x);
            }
        }
        None
    }
}

fn main() {}
