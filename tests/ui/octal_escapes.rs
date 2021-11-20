#![warn(clippy::octal_escapes)]

fn main() {
    let _bad1 = "\033[0m";
    let _bad2 = b"\033[0m";
    let _bad3 = "\\\033[0m";
    let _bad4 = "\01234567";
    let _bad5 = "\0\03";

    let _good1 = "\\033[0m";
    let _good2 = "\0\\0";
}
