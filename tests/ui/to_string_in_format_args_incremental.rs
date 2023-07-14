//@run-rustfix
//@compile-flags: -C incremental=target/debug/test/incr

// see https://github.com/rust-lang/rust-clippy/issues/10969

fn main() {
    let s = "Hello, world!";
    println!("{}", s.to_string());
}
