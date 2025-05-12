struct Wrapper<'a, T>(&'a T)
where
    T: 'a;

impl<'a, T> Drop for Wrapper<'a, T>
where
    T: 'static,
    //~^ error: `Drop` impl requires `T: 'static` but the struct it is implemented for does not
{
    fn drop(&mut self) {}
}

fn main() {}
