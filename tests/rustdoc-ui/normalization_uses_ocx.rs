// check-pass
// compile-flags: -Znormalize-docs
// regression test for #112242

trait MyTrait<'a> {
    type MyItem;
}
struct Inner<Q>(Q);
struct Outer<Q>(Inner<Q>);

unsafe impl<'a, Q> Send for Inner<Q>
where
    Q: MyTrait<'a>,
    <Q as MyTrait<'a>>::MyItem: Copy,
{
}
