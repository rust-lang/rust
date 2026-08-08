//@ known-bug: rust-lang/rust#144564
//@ needs-rustc-debug-assertions

trait Trait<'a> {
    type Out;
}

fn weird_bound<X>() -> X
where
    for<'a> X: Trait<'a>,
    <X as Trait<'a>>::Out: Copy,
{
    todo!()
}
