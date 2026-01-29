#![feature(supertrait_auto_impl)]

trait Super {}

trait Sub: Super {
    auto impl Super; // OK
}

trait SubBlock: Super {
    auto impl Super {} // OK
}

trait SubUnsafe: Super {
    unsafe auto impl Super; // OK
}

trait Super2 {
    type Type;
}

trait SubWithContent: Super2 {
    auto impl Super2 {
        type Type = u8; // OK
    }
}

struct Data<T>(T);

impl SubWithContent for Data<u8> {
    auto impl Super2 {
        type Type = u32; // OK
    }
}

impl SubWithContent for Data<u32> {
    extern impl Super2; // OK
}

impl SubWithContent for Data<u16> {
    unsafe extern impl Super2; // OK
}

// Negative tests
trait SubConstImpl: Super {
    const impl Super {}
    //~^ ERROR implementation is not supported in `trait`s or `impl`s
}
trait SubJustImpl: Super {
    impl Super {}
    //~^ ERROR implementation is not supported in `trait`s or `impl`s
}
