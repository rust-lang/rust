use std::fmt;

pub trait Bar {}

impl<'a> Bar + 'a {
    pub fn bar(&self) -> usize { 42 }
}

impl<'a> fmt::Debug for Bar + 'a {
    fn fmt(&self, _: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}
