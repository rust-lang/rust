trait SomeTrait {
    type Wrapped<A>: SomeTrait;

    fn f() -> ();
}

fn program<W>() -> ()
where
    W: SomeTrait<Wrapped = W>,
    //~^ ERROR: missing generics for associated type `SomeTrait::Wrapped`
{
    return W::f();
}

fn main() {}
