#![crate_type = "lib"]

pub trait U/*: ?Sized */ {
    fn modified(self) -> Self
    where
        Self: Sized
    {
        self
    }

    fn touch(&self)/* where Self: ?Sized */{}
}

pub trait S: Sized {}
