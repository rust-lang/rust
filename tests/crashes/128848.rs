//@ known-bug: rust-lang/rust#128848

fn f<T>(a: T, b: T, c: T)  {
    f.call_once()
}
