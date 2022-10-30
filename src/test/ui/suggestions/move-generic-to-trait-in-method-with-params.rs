// Generalizes the suggestion introduced in #100838

trait Foo<T> {
    fn bar(&self, _: T);
}

impl Foo<i32> for i32 {
    fn bar(&self, x: i32) {
        println!("{}", self + x);
    }
}

fn main() {
    1.bar::<i32>(0);
    //~^ ERROR this associated function takes 0 generic arguments but 1 generic argument was supplied
    //~| HELP consider moving this generic argument to the `Foo` trait, which takes up to 1 argument
    //~| HELP remove these generics
}
