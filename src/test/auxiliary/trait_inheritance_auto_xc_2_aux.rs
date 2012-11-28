pub trait Foo { fn f() -> int; }
pub trait Bar { fn g() -> int; }
pub trait Baz { fn h() -> int; }

pub struct A { x: int }

impl A : Foo { fn f() -> int { 10 } }
impl A : Bar { fn g() -> int { 20 } }
impl A : Baz { fn h() -> int { 30 } }


