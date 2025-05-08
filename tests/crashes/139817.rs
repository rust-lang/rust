//@ known-bug: #139817
fn enum_upvar() {
    type T = impl Copy;
    let foo: T = Some((42, std::marker::PhantomData::<T>));
    let x = move || match foo {
        None => (),
    };
}
