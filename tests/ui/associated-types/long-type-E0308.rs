//@ revisions: short shortest short-color shortest-color
//@[short-color] compile-flags: --diagnostic-width=100 --error-format=human --color=always -Zwrite-long-types-to-disk=yes
//@[shortest-color] compile-flags: --diagnostic-width=12 --error-format=human --color=always -Zwrite-long-types-to-disk=yes
//@[short] compile-flags: --diagnostic-width=100 -Zwrite-long-types-to-disk=yes
//@[shortest] compile-flags: --diagnostic-width=12 -Zwrite-long-types-to-disk=yes
//@[shortest-color] only-linux
//@[short-color] only-linux
// this does actually affect the diagnostic message
// even though it doesn't show up, separate issue ig
mod really_really_really_long_module {
    pub trait FooBar {
        type Assoc: FooBar;
        type Wrapper<T>: FooBar;

        fn assoc(&self) -> Self::Assoc;
        fn wrap<T>(&self, value: T) -> Self::Wrapper<T>;
    }
}

use really_really_really_long_module::*;

fn foo<'a, T: FooBar>(
    value: T
) -> <
    <
        <T::Wrapper<T> as FooBar>::Assoc as FooBar
    >::Assoc as FooBar
>::Wrapper<<T::Assoc as FooBar>::Assoc>
{
    // Ensure that we only highlight `FooBar::Wrapper` vs `FooBar::Assoc` in the E0308, instead of
    // the whole type for both expected and found.
    value.assoc().assoc().assoc().wrap(value.assoc().assoc())
    //[short]~^ ERROR E0308
    //[shortest]~^^ ERROR E0308
}

trait Bar<P> {
    type Assoc;
}

fn bar<T>(x: <T as Bar<u8>>::Assoc) -> <T as Bar<u16>>::Assoc
where
    T: Bar<u8> + Bar<u16>,
{
    // Ensure we highlight `u8` and `u16`.
    x
    //[short]~^ ERROR E0308
    //[shortest]~^^ ERROR E0308
}

trait Qux {
    type Assoc<K>;
}

fn qux<T>(x: <T as Qux>::Assoc<u8>) -> <T as Qux>::Assoc<u16>
where
    T: Qux,
{
    // Ensure we highlight `u8` and `u16`.
    x
    //[short]~^ ERROR E0308
    //[shortest]~^^ ERROR E0308
}

fn main() {}
