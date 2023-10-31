#[derive(Default)]
struct E<T> {
    data: T,
}

trait TestMut {
    type Output<'a>; //~ ERROR missing required bound
    fn test_mut<'a>(&'a mut self) -> Self::Output<'a>;
}

impl<T> TestMut for E<T>
where
    T: 'static,
{
    type Output<'a> = &'a mut T;
    fn test_mut<'a>(&'a mut self) -> Self::Output<'a> {
        &mut self.data
    }
}

fn test_simpler<'a>(dst: &'a mut impl TestMut<Output = &'a mut f32>)
  //~^ ERROR missing generics for associated type
{
    for n in 0i16..100 {
        *dst.test_mut() = n.into();
    }
}

fn main() {
    let mut t1: E<f32> = Default::default();
    test_simpler(&mut t1);
}
