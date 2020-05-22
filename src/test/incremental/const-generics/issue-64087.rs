// revisions:cfail1
#![feature(const_generics)]
//[cfail1]~^ WARN the feature `const_generics` is incomplete

fn combinator<T, const S: usize>() -> [T; S] {}
//[cfail1]~^ ERROR mismatched types

fn main() {
    combinator().into_iter();
    //[cfail1]~^ ERROR type annotations needed
}
