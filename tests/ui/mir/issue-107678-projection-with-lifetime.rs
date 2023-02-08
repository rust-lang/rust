// build-pass

#![crate_type = "lib"]

pub trait StreamOnce {
    type Error;
}

pub trait ResetStream: StreamOnce {
    fn reset(&mut self) -> Result<(), Self::Error>;
}

impl<'a> ResetStream for &'a str
    where Self: StreamOnce
{
    #[inline]
    fn reset(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}
