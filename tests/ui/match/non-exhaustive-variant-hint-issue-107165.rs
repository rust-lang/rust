//@ aux-build:non_exhaustive_structs_and_variants_lib.rs

/* Provide a hint to the user that a match of a non-exhaustive variant might
 * fail because of a missing struct pattern (issue #107165).
 */

// Ignore non_exhaustive in the same crate
enum Elocal {
    #[non_exhaustive]
    Unit,

    #[non_exhaustive]
    Tuple(i64),
}

extern crate non_exhaustive_structs_and_variants_lib;
use non_exhaustive_structs_and_variants_lib::Elibrary;

fn local() -> Elocal {
    todo!()
}

fn library() -> Elibrary {
    todo!()
}

fn main() {
    let loc = local();
    // No error for enums defined in this crate
    match loc {
        Elocal::Unit => (),
        Elocal::Tuple(_) => (),
    };

    // Elibrary is externally defined
    let lib = library();

    match lib {
        Elibrary::Unit => (),
        //~^ ERROR unit variant `Unit` is private [E0603]
        _ => (),
    };

    match lib {
        Elibrary::Tuple(_) => (),
        //~^ ERROR tuple variant `Tuple` is private [E0603]
        _ => (),
    };
}
