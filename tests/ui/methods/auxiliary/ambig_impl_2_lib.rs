pub trait Me {
    fn me(&self) -> usize;
}
impl Me for usize { fn me(&self) -> usize { *self } }
