// compile-flags: -Ztrait-solver=next
// check-pass

#![feature(return_position_impl_trait_in_trait)]

trait Foo {
    fn test() -> impl Fn(u32) -> u32 {
        |x| x.count_ones()
    }
}

fn main() {}
