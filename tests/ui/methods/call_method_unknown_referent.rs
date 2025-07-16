//@ edition: 2018

#![feature(arbitrary_self_types)]

// tests that the referent type of a reference must be known to call methods on it

struct SmartPtr<T>(T);

impl<T> core::ops::Receiver for SmartPtr<T> {
    type Target = T;
}

impl<T> SmartPtr<T> {
    fn foo(&self) {}
}

fn main() {
    let val = 1_u32;
    let ptr = &val;
    let _a: i32 = (ptr as &_).read();
    //~^ ERROR type annotations needed

    // Same again, but with a smart pointer type
    let val2 = 1_u32;
    let rc = std::rc::Rc::new(val2);
    let _b = (rc as std::rc::Rc<_>).read();
    //~^ ERROR type annotations needed

    // Same again, but with a smart pointer type
    let ptr = SmartPtr(val);

    // We can call unambiguous outer-type methods on this
    (ptr as SmartPtr<_>).foo();
    // ... but we can't follow the Receiver chain to the inner type
    // because we end up with _.

    // Because SmartPtr implements Receiver, it's arguable which of the
    // following two diagnostics we'd want in this case:
    // (a) "type annotations needed" (because the inner type is _)
    // (b) "no method named `read` found for struct `SmartPtr`"
    //     (ignoring the fact that there might have been methods on the
    //      inner type, had it not been _)
    // At present we produce error type (b), which is necessary because
    // our resolution logic needs to be able to call methods such as foo()
    // on the outer type even if the inner type is ambiguous.
    let _c = (ptr as SmartPtr<_>).read();
    //~^ ERROR no method named `read` found for struct `SmartPtr<T>`
}
