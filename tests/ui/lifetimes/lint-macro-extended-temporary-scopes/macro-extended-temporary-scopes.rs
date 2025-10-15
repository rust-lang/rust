//! Future-compatibility warning test for #145838: make sure we catch all expected breakage.
//! Shortening temporaries in the tails of block expressions should warn in Rust 2024, and
//! shortening temporaries in the tails of `if` expressions' blocks should warn in all editions.
//@ revisions: e2021 e2024
//@ [e2021] edition: 2021
//@ [e2024] edition: 2024
//@ check-pass
use std::pin::pin;

struct Struct { field: () }

fn cond() -> bool { true }
fn temp() {}
fn array_temp() -> [(); 1] { [()] }
fn tuple_temp() -> ((),) { ((),) }
fn struct_temp() -> Struct { Struct { field: () } }
fn smart_ptr_temp() -> Box<()> { Box::new(()) }

const CONST_STRING: String = String::new();
static STATIC_UNIT: () = ();

fn main() {
    let local = String::new();

    // #145880 doesn't apply here, so this `temp()`'s lifetime is reduced by #145838 in Rust 2024.
    println!("{:?}{:?}", { &temp() }, ());
    //[e2024]~^ WARN temporary lifetime will be shortened in Rust 1.92
    //[e2024]~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

    // In real-world projects, this breakage typically appeared in `if` expressions with a reference
    // to a `String` temporary in one branch's tail expression. This is edition-independent since
    // `if` expressions' blocks are temporary scopes in all editions.
    println!("{:?}{:?}", (), if cond() { &format!("") } else { "" });
    //~^ WARN temporary lifetime will be shortened in Rust 1.92
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    println!("{:?}{:?}", (), if cond() { &"".to_string() } else { "" });
    //~^ WARN temporary lifetime will be shortened in Rust 1.92
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    println!("{:?}{:?}", (), if cond() { &("string".to_owned() + "string") } else { "" });
    //~^ WARN temporary lifetime will be shortened in Rust 1.92
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

    // Make sure we catch indexed and dereferenced temporaries.
    pin!(
        if cond() {
            &array_temp()[0]
            //~^ WARN temporary lifetime will be shortened in Rust 1.92
            //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
        } else if cond() {
            &tuple_temp().0
            //~^ WARN temporary lifetime will be shortened in Rust 1.92
            //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
        } else if cond() {
            &struct_temp().field
            //~^ WARN temporary lifetime will be shortened in Rust 1.92
            //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
        } else {
            &*&*smart_ptr_temp()
            //~^ WARN temporary lifetime will be shortened in Rust 1.92
            //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
        }
    );

    // Test that `super let` extended by parent `super let`s in non-extending blocks are caught.
    pin!(pin!({ &temp() }));
    //[e2024]~^ WARN temporary lifetime will be shortened in Rust 1.92
    //[e2024]~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

    // We shouldn't warn when lifetime extension applies.
    let _ = format_args!("{:?}{:?}", { &temp() }, if cond() { &temp() } else { &temp() });
    let _ = pin!(
        if cond() {
            &array_temp()[0]
        } else if cond() {
            &tuple_temp().0
        } else if cond() {
            &struct_temp().field
        } else {
            &*&*smart_ptr_temp()
        }
    );
    let _ = pin!(pin!({ &temp() }));

    // We shouldn't warn when borrowing from non-temporary places.
    pin!({ &local });
    pin!({ &STATIC_UNIT });

    // We shouldn't warn for promoted constants.
    pin!({ &size_of::<()>() });
    pin!({ &(1 / 1) });
    pin!({ &mut ([] as [(); 0]) });
    pin!({ &None::<String> });
    pin!({ &|| String::new() });

    // But we do warn on these temporaries, since they aren't promoted.
    pin!({ &(1 / 0) });
    //[e2024]~^ WARN temporary lifetime will be shortened in Rust 1.92
    //[e2024]~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    pin!({ &mut [()] });
    //[e2024]~^ WARN temporary lifetime will be shortened in Rust 1.92
    //[e2024]~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    pin!({ &Some(String::new()) });
    //[e2024]~^ WARN temporary lifetime will be shortened in Rust 1.92
    //[e2024]~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    pin!({ &(|| ())() });
    //[e2024]~^ WARN temporary lifetime will be shortened in Rust 1.92
    //[e2024]~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    pin!({ &|| &local });
    //[e2024]~^ WARN temporary lifetime will be shortened in Rust 1.92
    //[e2024]~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    pin!({ &CONST_STRING });
    //[e2024]~^ WARN temporary lifetime will be shortened in Rust 1.92
    //[e2024]~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

    // This lint only catches future errors. Future dangling pointers do not produce warnings.
    pin!({ &raw const *&temp() });
}
