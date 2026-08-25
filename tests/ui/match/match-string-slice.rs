//! Regression test for <https://github.com/rust-lang/rust/issues/4541>.
//! This used to segfault caused by double-free.
//@ run-pass

fn parse_args() -> String {
    let args: Vec<_> = ::std::env::args().collect();
    let mut n = 0;

    while n < args.len() {
        match &*args[n] {
            "-v" => (),
            s => {
                return s.to_string();
            }
        }
        n += 1;
    }

    return "".to_string()
}

pub fn main() {
    println!("{}", parse_args());
}
