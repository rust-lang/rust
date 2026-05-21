pub trait Parker {
    type Interrupt;

    fn park(self, validate: impl FnOnce(usize) -> bool) -> Result<(), Self::Interrupt>;
}

pub trait Unparker: Copy {
    fn unpark(self, thread_index: usize);
}
