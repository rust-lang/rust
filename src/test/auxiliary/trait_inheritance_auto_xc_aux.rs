trait Foo { fn f() -> int; }
trait Bar { fn g() -> int; }
trait Baz { fn h() -> int; }

trait Quux: Foo Bar Baz { }

impl<T: Foo Bar Baz> T: Quux { }
