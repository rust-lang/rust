// compile-pass
// compile-flags: -Z parse-only

type A = Box<(Fn(D::Error) -> E) + 'static + Send + Sync>; // OK (but see #39318)
