trait Foo {
    //~^ ERROR cycle detected
    type Context<'c>
    where
        Self: 'c;
}
//@ ignore-parallel-frontend query cycle
impl Foo for Box<dyn Foo> {}

fn main() {}
