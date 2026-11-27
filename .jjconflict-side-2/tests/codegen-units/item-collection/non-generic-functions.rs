//@ compile-flags:-Clink-dead-code

#![deny(dead_code)]
#![crate_type = "lib"]

//~ MONO_ITEM fn foo
fn foo() {
    {
        //~ MONO_ITEM fn foo::foo
        fn foo() {}
        foo();
    }

    {
        //~ MONO_ITEM fn foo::foo
        fn foo() {}
        foo();
    }
}

//~ MONO_ITEM fn bar
fn bar() {
    //~ MONO_ITEM fn bar::baz
    fn baz() {}
    baz();
}

struct Struct {
    _x: i32,
}

impl Struct {
    //~ MONO_ITEM fn Struct::foo
    fn foo() {
        {
            //~ MONO_ITEM fn Struct::foo::foo
            fn foo() {}
            foo();
        }

        {
            //~ MONO_ITEM fn Struct::foo::foo
            fn foo() {}
            foo();
        }
    }

    //~ MONO_ITEM fn Struct::bar
    fn bar(&self) {
        {
            //~ MONO_ITEM fn Struct::bar::foo
            fn foo() {}
            foo();
        }

        {
            //~ MONO_ITEM fn Struct::bar::foo
            fn foo() {}
            foo();
        }
    }
}

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    foo();
    bar();
    Struct::foo();
    let x = Struct { _x: 0 };
    x.bar();

    0
}
