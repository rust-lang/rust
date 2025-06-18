pub trait SizedTr {}

impl<T: Sized> SizedTr for T {}

pub trait NegSizedTr {}

impl<T: ?Sized> NegSizedTr for T {}
