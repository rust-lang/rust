// compile-flags: -Znext-solver
// check-pass

trait Foo {
    fn test() -> impl Fn(u32) -> u32 {
        |x| x.count_ones()
    }
}

fn main() {}
