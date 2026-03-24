//@ revisions:cfail1

fn combinator<T, const S: usize>() -> [T; S] {}
//[cfail1]~^ ERROR mismatched types

fn main() {
    combinator().into_iter();
    //[cfail1]~^ ERROR type annotations needed
    //[cfail1]~| ERROR type annotations needed
    //[cfail1]~| ERROR type annotations needed
}
