trait Foo = Default;
//~^ ERROR trait aliases are experimental

macro_rules! accept_item {
    ($i:item) => {}
}

accept_item! {
    trait Foo = Ord + Eq;
    //~^ ERROR trait aliases are experimental
}

fn main() {}
