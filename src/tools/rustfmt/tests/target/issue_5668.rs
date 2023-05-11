type Foo = impl Send;
struct Struct<
    const C: usize = {
        let _: Foo = ();
        //~^ ERROR: mismatched types
        0
    },
>;
