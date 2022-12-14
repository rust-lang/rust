#![feature(auto_traits)]

auto trait Trait1<'a> {}
//~^ ERROR auto traits cannot have generic parameters

fn f<'a>(x: &dyn Trait1<'a>)
{}

fn main() {
    f(&1);
}
