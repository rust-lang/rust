pub struct DefaultLifetime<'a, 'b = 'static> {
                                   //~^ ERROR unexpected default lifetime parameter
    _marker: std::marker::PhantomData<&'a &'b ()>,
}

fn main(){}
