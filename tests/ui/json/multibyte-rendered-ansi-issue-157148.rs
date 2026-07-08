// Regression test for <https://github.com/rust-lang/rust/issues/157148>.
// JSON rendered diagnostics should not ICE when spans involve non-ASCII source text.

//@ edition: 2021
//@ check-pass
//@ compile-flags: --error-format=json --json=diagnostic-rendered-ansi
//@ normalize-stderr: "(\\u001b\[[0-9;]+m)+" -> "[ANSI]"

#![warn(unused_mut)]

fn main() {
    let mut x = 0; // 这是一段中文注释，用于在被下划线标注的行中加入多字节字符
    //~^ WARNING variable does not need to be mutable
    //~| HELP remove this `mut`
    let _ = x;
}
