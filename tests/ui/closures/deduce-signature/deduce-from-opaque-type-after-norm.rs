//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass
trait Foo {
    fn test() -> impl Fn(u32) -> u32 {
        |x| x.count_ones()
    }
}

fn main() {}
