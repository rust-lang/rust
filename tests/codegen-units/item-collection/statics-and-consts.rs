//@ compile-flags:-Clink-dead-code

#![deny(dead_code)]
#![crate_type = "lib"]

static STATIC1: i64 = {
    const STATIC1_CONST1: i64 = 2;
    1 + CONST1 as i64 + STATIC1_CONST1
};

const CONST1: i64 = {
    const CONST1_1: i64 = {
        const CONST1_1_1: i64 = 2;
        CONST1_1_1 + 1
    };
    1 + CONST1_1 as i64
};

fn foo() {
    let _ = {
        const CONST2: i64 = 0;
        static STATIC2: i64 = CONST2;

        let x = {
            const CONST2: i64 = 1;
            static STATIC2: i64 = CONST2;
            STATIC2
        };

        x + STATIC2
    };

    let _ = {
        const CONST2: i64 = 0;
        static STATIC2: i64 = CONST2;
        STATIC2
    };
}

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    foo();
    let _ = STATIC1;

    0
}

//~ MONO_ITEM static STATIC1

//~ MONO_ITEM fn foo
//~ MONO_ITEM static foo::STATIC2
//~ MONO_ITEM static foo::STATIC2
//~ MONO_ITEM static foo::STATIC2
