// compile-flags: -Cmetadata=aux

pub trait FooAux {
    #[doc(hidden)]
    fn foo(&self) {}

    #[doc(hidden)]
    fn foo2(&self);
}

impl FooAux for i32 {
  fn foo2(&self) {}
}
