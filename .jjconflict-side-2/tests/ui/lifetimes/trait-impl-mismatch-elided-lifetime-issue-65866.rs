// Regression test for https://github.com/rust-lang/rust/issues/65866.

mod plain {
    struct Foo;

    struct Re<'a> {
        _data: &'a u16,
    }

    trait Bar {
        fn bar(&self, r: &mut Re);
        //~^ NOTE expected
        //~| NOTE `Re` here is elided as `Re<'_>`
    }

    impl Bar for Foo {
        fn bar<'a, 'b>(&'a self, _r: &'b mut Re<'a>) {}
        //~^ ERROR `impl` item signature doesn't match `trait` item signature
        //~| NOTE expected signature
        //~| NOTE found
        //~| HELP the lifetime requirements
        //~| HELP verify the lifetime relationships
    }
}

mod with_type_args {
    struct Foo;

    struct Re<'a, T> {
        _data: (&'a u16, T),
    }

    trait Bar {
        fn bar(&self, r: &mut Re<u8>);
        //~^ NOTE expected
        //~| NOTE `Re` here is elided as `Re<'_, u8>`
    }

    impl Bar for Foo {
        fn bar<'a, 'b>(&'a self, _r: &'b mut Re<'a, u8>) {}
        //~^ ERROR `impl` item signature doesn't match `trait` item signature
        //~| NOTE expected signature
        //~| NOTE found
        //~| HELP the lifetime requirements
        //~| HELP verify the lifetime relationships
    }
}

fn main() {}
