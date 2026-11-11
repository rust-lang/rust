pub struct P;
pub struct Map<F>(pub F);

pub trait PIter: Sized {
    fn map<F>(self, f: F) -> Map<F>
    where
        F: Fn(i32);

    fn flat_map<F, U>(self, f: F)
    where
        F: Fn(i32) -> U;
}

impl PIter for P {
    fn map<F>(self, f: F) -> Map<F>
    where
        F: Fn(i32),
    {
        Map(f)
    }

    fn flat_map<F, U>(self, _: F)
    where
        F: Fn(i32) -> U,
    {
    }
}

pub fn to_fn<F>(f: F) -> Map<F>
where
    F: Fn(i32),
{
    Map(f)
}
