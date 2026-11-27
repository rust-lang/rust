#![feature(auto_traits)]

auto trait Trait1<'outer> {}
//~^ ERROR auto traits cannot have generic parameters

fn f<'a>(x: impl Trait1<'a>) {}

fn main() {
    f("");
}
