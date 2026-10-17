pub fn sized_param<T>() {}

pub fn relaxed_sized_on_param<T: ?Sized>() {}

pub trait SizedOnParentParam<T: ?Sized> {
    fn func() where T: Sized;
}

pub trait SizedSelf: Sized {}

pub trait SizedOnParentSelf {
    fn func(self) -> Self
    where
        Self: Sized
    {
        self
    }
}
