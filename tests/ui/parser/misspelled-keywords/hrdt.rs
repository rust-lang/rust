struct Closure<F> {
    data: (u8, u16),
    func: F,
}

impl<F> Closure<F>
    Where for<'a> F: Fn(&'a (u8, u16)) -> &'a u8,
//~^ ERROR expected one of
{
    fn call(&self) -> &u8 {
        (self.func)(&self.data)
    }
}


fn main() {}
