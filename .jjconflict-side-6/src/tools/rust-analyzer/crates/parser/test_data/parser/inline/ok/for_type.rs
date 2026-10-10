type A = for<'a> fn() -> ();
type B = for<'a> unsafe extern "C" fn(&'a ()) -> ();
type Obj = for<'a> PartialEq<&'a i32>;
