#![deny(unused_must_use)]

fn foo(_: i32) -> bool { todo!() }

fn bar() -> &'static i32 {
    &42;
    //~^ ERROR unused

    &mut foo(42);
    //~^ ERROR unused

    &&42;
    //~^ ERROR unused

    &&mut 42;
    //~^ ERROR unused

    &mut &42;
    //~^ ERROR unused

    let _result = foo(4)
        && foo(2); // Misplaced semi-colon (perhaps due to reordering of lines)
    && foo(42);
    //~^ ERROR unused

    let _ = &42; // ok

    &42 // ok
}

fn main() {
    let _ = bar();
}
