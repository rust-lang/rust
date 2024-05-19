// Regression test for issue #1448 and #1386

fn foo(a: u32) -> u32 { a }

fn main() {
    println!("{}", foo(10i32)); //~ ERROR mismatched types
}
