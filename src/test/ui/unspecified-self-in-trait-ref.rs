pub trait Foo<A=Self> {
    fn foo(&self);
}

pub trait Bar<X=usize, A=Self> {
    fn foo(&self);
}

fn main() {
    let a = Foo::lol();
    //~^ ERROR no function or associated item named
    let b = Foo::<_>::lol();
    //~^ ERROR no function or associated item named
    let c = Bar::lol();
    //~^ ERROR no function or associated item named
    let d = Bar::<usize, _>::lol();
    //~^ ERROR no function or associated item named
    let e = Bar::<usize>::lol();
    //~^ ERROR must be explicitly specified
}
