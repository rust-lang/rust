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
    // TODO: warn in Rust 2024

    // In real-world projects, this breakage typically appeared in `if` expressions with a reference
    // to a `String` temporary in one branch's tail expression. This is edition-independent since
    // `if` expressions' blocks are temporary scopes in all editions.
    println!("{:?}{:?}", (), if cond() { &format!("") } else { "" });
    // TODO: warn in all editions
    println!("{:?}{:?}", (), if cond() { &"".to_string() } else { "" });
    // TODO: warn in all editions
    println!("{:?}{:?}", (), if cond() { &("string".to_owned() + "string") } else { "" });
    // TODO: warn in all editions

    // Make sure we catch indexed and dereferenced temporaries.
    pin!(
        if cond() {
            &array_temp()[0]
            // TODO: warn in all editions
        } else if cond() {
            &tuple_temp().0
            // TODO: warn in all editions
        } else if cond() {
            &struct_temp().field
            // TODO: warn in all editions
        } else {
            &*&*smart_ptr_temp()
            // TODO: warn in all editions
        }
    );

    // Test that `super let` extended by parent `super let`s in non-extending blocks are caught.
    pin!(pin!({ &temp() }));
    // TODO: warn in Rust 2024

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
    // TODO: warn in Rust 2024
    pin!({ &mut [()] });
    // TODO: warn in Rust 2024
    pin!({ &Some(String::new()) });
    // TODO: warn in Rust 2024
    pin!({ &(|| ())() });
    // TODO: warn in Rust 2024
    pin!({ &|| &local });
    // TODO: warn in Rust 2024
    pin!({ &CONST_STRING });
    // TODO: warn in Rust 2024

    // This lint only catches future errors. Future dangling pointers do not produce warnings.
    pin!({ &raw const *&temp() });
}
