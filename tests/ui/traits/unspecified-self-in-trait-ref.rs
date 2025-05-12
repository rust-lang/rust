pub trait Foo<A=Self> {
    fn foo(&self);
}

pub trait Bar<X=usize, A=Self> {
    fn foo(&self);
}

fn main() {
    let a = Foo::lol();
    //~^ ERROR no function or associated item named
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
    let b = Foo::<_>::lol();
    //~^ ERROR no function or associated item named
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
    let c = Bar::lol();
    //~^ ERROR no function or associated item named
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
    let d = Bar::<usize, _>::lol();
    //~^ ERROR no function or associated item named
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
    let e = Bar::<usize>::lol();
    //~^ ERROR must be explicitly specified
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
}
