pub trait MyTrait {
    type Assoc;
}

pub fn foo<S, T>(_s: S, _t: T)
where
    S: MyTrait,
    T: MyTrait<Assoc == S::Assoc>,
    //~^ ERROR: expected one of `,` or `>`, found `==`
    //~| ERROR: trait takes 0 generic arguments but 1 generic argument was supplied
{
}

fn main() {}
