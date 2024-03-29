#![feature(c_variadic)]
#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod generics {
    trait GenericTrait<T> {
        fn bar(&self, x: T) -> T { x }
        fn bar1() {}
    }
    trait Trait {
        fn foo(&self, x: i32) -> i32 { x }
        fn foo1<'a>(&self, x: &'a i32) -> &'a i32 { x }
        fn foo2<T>(&self, x: T) -> T { x }
        fn foo3<'a: 'a>(_: &'a u32) {}

        reuse GenericTrait::bar;
        //~^ ERROR early bound generics are not supported for associated delegation items
        reuse GenericTrait::bar1;
        //~^ ERROR early bound generics are not supported for associated delegation items
    }

    struct F;
    impl Trait for F {}
    impl<T> GenericTrait<T> for F {}

    struct S(F);

    impl<T> GenericTrait<T> for S {
        reuse <F as GenericTrait<T>>::bar { &self.0 }
        //~^ ERROR early bound generics are not supported for associated delegation items
        reuse GenericTrait::<T>::bar1;
        //~^ ERROR early bound generics are not supported for associated delegation items
    }

    impl GenericTrait<()> for () {
        reuse GenericTrait::bar { &F }
        //~^ ERROR early bound generics are not supported for associated delegation items
        reuse GenericTrait::bar1;
        //~^ ERROR early bound generics are not supported for associated delegation items
    }

    impl Trait for &S {
        reuse Trait::foo;
        //~^ ERROR early bound generics are not supported for associated delegation items
    }

    impl Trait for S {
        reuse Trait::foo1 { &self.0 }
        reuse Trait::foo2 { &self.0 }
        //~^ ERROR early bound generics are not supported for associated delegation items
        //~| ERROR method `foo2` has 0 type parameters but its trait declaration has 1 type parameter
        reuse <F as Trait>::foo3;
        //~^ ERROR early bound generics are not supported for associated delegation items
        //~| ERROR lifetime parameters or bounds on method `foo3` do not match the trait declaration
    }

    struct GenericS<T>(T);
    impl<T> Trait for GenericS<T> {
        reuse Trait::foo { &self.0 }
        //~^ ERROR early bound generics are not supported for associated delegation items
    }
}

mod opaque {
    trait Trait {}
    impl Trait for () {}

    mod to_reuse {
        use super::Trait;

        pub fn opaque_arg(_: impl Trait) -> i32 { 0 }
        pub fn opaque_ret() -> impl Trait { unimplemented!() }
    }
    reuse to_reuse::opaque_arg;

    trait ToReuse {
        fn opaque_ret() -> impl Trait { unimplemented!() }
    }

    // FIXME: Inherited `impl Trait`s create query cycles when used inside trait impls.
    impl ToReuse for u8 {
        reuse to_reuse::opaque_ret; //~ ERROR cycle detected when computing type
    }
    impl ToReuse for u16 {
        reuse ToReuse::opaque_ret; //~ ERROR cycle detected when computing type
    }
}

mod recursive {
    mod to_reuse1 {
        pub mod to_reuse2 {
            pub fn foo() {}
        }

        pub reuse to_reuse2::foo;
    }

    reuse to_reuse1::foo;
    //~^ ERROR recursive delegation is not supported yet
}

fn main() {}
