#![feature(min_generic_const_args, adt_const_params)]
#![expect(incomplete_features)]

#[derive(Eq, PartialEq, std::marker::ConstParamTy)]
enum Option<T> {
    Some(T),
    None,
}
use Option::Some;

fn foo<const N: Option<u32>>() {}

trait Trait {
    #[type_const]
    const ASSOC: u32;
}

fn bar<T: Trait, const N: u32>() {
    // the initializer of `_0` is a `N` which is a legal const argument
    // so this is ok.
    foo::<{ Some::<u32> { 0: N } }>();

    // this is allowed as mgca supports uses of assoc consts in the
    // type system. ie `<T as Trait>::ASSOC` is a legal const argument
    foo::<{ Some::<u32> { 0: <T as Trait>::ASSOC } }>();

    // this on the other hand is not allowed as `N + 1` is not a legal
    // const argument
    foo::<{ Some::<u32> { 0: N + 1 } }>();
    //~^ ERROR: complex const arguments must be placed inside of a `const` block

    // this also is not allowed as generic parameters cannot be used
    // in anon const const args
    foo::<{ Some::<u32> { 0: const { N + 1 } } }>();
    //~^ ERROR: generic parameters may not be used in const operations
}

fn main() {}
