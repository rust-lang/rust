// compile-flags: -Znext-solver
// FIXME(-Znext-solver): This test is currently broken because the `deduce_closure_signature`
// is unable to look at nested obligations.
trait Foo {
    fn test() -> impl Fn(u32) -> u32 {
        |x| x.count_ones()
        //~^ ERROR type annotations needed
    }
}

fn main() {}
