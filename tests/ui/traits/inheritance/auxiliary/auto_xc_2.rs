pub trait Foo { fn f(&self) -> isize; }
pub trait Bar { fn g(&self) -> isize; }
pub trait Baz { fn h(&self) -> isize; }

pub struct A { pub x: isize }

impl Foo for A { fn f(&self) -> isize { 10 } }
impl Bar for A { fn g(&self) -> isize { 20 } }
impl Baz for A { fn h(&self) -> isize { 30 } }
