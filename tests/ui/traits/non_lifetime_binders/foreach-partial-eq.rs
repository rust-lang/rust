#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

fn auto_trait()
where
    for<T> T: PartialEq + PartialOrd,
{}

fn main() {
    auto_trait();
    //~^ ERROR can't compare `T` with `T`
}
