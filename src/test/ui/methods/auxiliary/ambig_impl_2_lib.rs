pub trait me {
    fn me(&self) -> usize;
}
impl me for usize { fn me(&self) -> usize { *self } }
