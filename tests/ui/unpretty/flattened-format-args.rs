// compile-flags: -Zunpretty=hir -Zflatten-format-args=yes
// check-pass

fn main() {
    let x = 1;
    // Should flatten to println!("a 123 b {x} xyz\n"):
    println!("a {} {}", format_args!("{} b {x}", 123), "xyz");
}
