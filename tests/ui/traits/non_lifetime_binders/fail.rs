// Error reporting for where `for<T> T: Trait` doesn't hold

#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

trait Trait {}

fn fail()
where
    for<T> T: Trait,
{}

fn auto_trait()
where
    for<T> T: Send,
{}

fn main() {
    fail();
    //~^ ERROR the trait bound `T: Trait` is not satisfied
    auto_trait();
    //~^ ERROR `T` cannot be sent between threads safely
}
