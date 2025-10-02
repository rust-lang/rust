//@ compile-flags: -Znext-solver
trait Super<'a> {}
trait Trait<'a>: Super<'a> + for<'hr> Super<'hr> {}

fn foo<'a>(x: Box<dyn Trait<'a>>) -> Box<dyn Super<'a>> {
    x
    //~^ ERROR type annotations needed: cannot satisfy `dyn Trait<'_>: Unsize<dyn Super<'_>>
}

fn main() {}
