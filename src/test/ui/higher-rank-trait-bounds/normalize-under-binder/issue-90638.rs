//check-pass

trait Yokeable<'a>: 'static {
    type Output: 'a;
}

trait IsCovariant<'a> {}

struct Yoke<Y: for<'a> Yokeable<'a>> {
    data: Y,
}

impl<Y: for<'a> Yokeable<'a>> Yoke<Y> {
    fn project<Y2: for<'a> Yokeable<'a>>(&self, _f: for<'a> fn(<Y as Yokeable<'a>>::Output, &'a ())
      -> <Y2 as Yokeable<'a>>::Output) -> Yoke<Y2> {

        unimplemented!()
    }
}

fn _upcast<Y>(x: Yoke<Y>) -> Yoke<Box<dyn IsCovariant<'static> + 'static>> where
    Y: for<'a> Yokeable<'a>,
    for<'a> <Y as Yokeable<'a>>::Output: IsCovariant<'a>
    {
    x.project(|data, _| {
        Box::new(data)
    })
}


impl<'a> Yokeable<'a> for Box<dyn IsCovariant<'static> + 'static> {
    type Output = Box<dyn IsCovariant<'a> + 'a>;
}

fn main() {}
