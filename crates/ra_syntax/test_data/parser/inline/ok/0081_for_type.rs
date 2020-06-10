type A = for<'a> fn() -> ();
type B = for<'a> unsafe extern "C" fn(&'a ()) -> ();
