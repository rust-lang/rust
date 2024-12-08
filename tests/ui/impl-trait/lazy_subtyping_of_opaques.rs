//! This test checks that we allow subtyping predicates that contain opaque types.
//! No hidden types are being constrained in the subtyping predicate, but type and
//! lifetime variables get subtyped in the generic parameter list of the opaque.

//@ check-pass

fn foo() -> impl Default + Copy {
    if false {
        let x = Default::default();
        // add `Subtype(?x, ?y)` obligation
        let y = x;

        // Make a tuple `(?x, ?y)` and equate it with `(impl Default, u32)`.
        // For us to try and prove a `Subtype(impl Default, u32)` obligation,
        // we have to instantiate both `?x` and `?y` without any
        // `select_where_possible` calls inbetween.
        let mut tup = &mut (x, y);
        let assign_tup = &mut (foo(), 1u32);
        tup = assign_tup;
    }
    1u32
}

fn main() {}
