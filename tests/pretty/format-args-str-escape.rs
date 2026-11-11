//@ pretty-compare-only
//@ pretty-mode:expanded
//@ pp-exact:format-args-str-escape.pp

fn main() {
    println!("\x1B[1mHello, world!\x1B[0m");
    println!("\u{1B}[1mHello, world!\u{1B}[0m");
    println!("Not an escape sequence: \\u{{1B}}[1mbold\\x1B[0m");
    println!("{}", "\x1B[1mHello, world!\x1B[0m");
}
