trait A: B + A {}
//~^ ERROR cycle detected when computing the super predicates of `A`. see https://rustc-dev-guide.rust-lang.org/overview.html#queries and https://rustc-dev-guide.rust-lang.org/query.html for more information. [E0391]

trait B {}

impl A for () {}

impl B for () {}

fn main() {
    let a: Box<dyn A> = Box::new(());
    let _b: Box<dyn B> = a;
}
