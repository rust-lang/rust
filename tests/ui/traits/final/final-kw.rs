//@ revisions: ungated gated

#[cfg(ungated)]
trait Trait {
    final fn foo() {}
    //~^ ERROR `final` on trait functions is experimental
}

fn main() {}
