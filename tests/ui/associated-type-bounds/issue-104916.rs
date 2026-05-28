trait B {
    type AssocType;
}

fn f()
where
    dyn for<'j> B<AssocType: 'j>:,
    //~^ ERROR: associated type bounds are not allowed in `dyn` types
{
}

fn main() {}
