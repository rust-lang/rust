//@ revisions: bfail1

fn combinator<T, const S: usize>() -> [T; S] {}
//[bfail1]~^ ERROR mismatched types

fn main() {
    combinator().into_iter();
    //[bfail1]~^ ERROR type annotations needed
    //[bfail1]~| ERROR type annotations needed
    //[bfail1]~| ERROR type annotations needed
}
