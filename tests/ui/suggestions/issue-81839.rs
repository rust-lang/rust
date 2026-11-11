//@ aux-build:issue-81839.rs
//@ edition:2018

extern crate issue_81839;

async fn test(ans: &str, num: i32, cx: &issue_81839::Test) -> u32 {
    match num {
        1 => {
            cx.answer_str("hi");
        }
        _ => cx.answer_str("hi"), //~ ERROR `match` arms have incompatible types
    }

    1
}

fn main() {}
