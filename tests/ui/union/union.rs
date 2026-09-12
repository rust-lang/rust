//@ aux-build: zst-const.rs

extern crate zst_const;
use zst_const::{
    HAS_NON_EXHAUSTIVE_FIELD_LIST, HAS_NON_EXHAUSTIVE_TUPLE_FIELD_LIST, HAS_PRIVATE_FIELD,
    HAS_PRIVATE_TUPLE_FIELD, HasNonExhaustiveFieldList, HasNonExhaustiveTupleFieldList,
    HasPrivateField, HasPrivateTupleField, MixedVisibilityUnion, NESTED_CONST,
};

union Foo {
    bar: i8,
    zst: (),
    tuple: (i32,),
    pizza: Pizza,
    tuple_struct: TupleStruct,
    array: [u32; 2],
    single_variant_enum: SingleVariant,
    has_private_field: HasPrivateField,
    has_non_exhaustive_field_list: HasNonExhaustiveFieldList,
    has_private_tuple_field: HasPrivateTupleField,
    has_non_exhaustive_tuple_field_list: HasNonExhaustiveTupleFieldList,
    local_non_exhaustive_field_list: LocalNonExhaustiveFieldList,
    nested_const: (HasPrivateField,),
    mixed_visibility_union: MixedVisibilityUnion,
}

#[derive(Clone, Copy)]
struct Pizza {
    topping: Option<PizzaTopping>,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum PizzaTopping {
    Cheese,
    Pineapple,
}

#[derive(Clone, Copy)]
struct TupleStruct(i32);

#[derive(Clone, Copy)]
enum SingleVariant {
    Single {},
}

#[derive(Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct LocalNonExhaustiveFieldList {}

pub const LOCAL_NON_EXHAUSTIVE_FIELD_LIST: LocalNonExhaustiveFieldList =
    LocalNonExhaustiveFieldList {};

fn do_nothing(_x: &mut Foo) {}

const UNIT: () = ();

pub fn main() {
    let mut foo = Foo { bar: 5 };
    do_nothing(&mut foo);

    // This is UB, so this test isn't run
    match foo {
        Foo { bar: _a } => {} //~ ERROR access to union field is unsafe
    }
    match foo {
        Foo {
            pizza:
                Pizza {
                    topping: Some(PizzaTopping::Cheese) | Some(PizzaTopping::Pineapple) | None,
                    //~^ ERROR access to union field is unsafe
                    //~| ERROR access to union field is unsafe
                    //~| ERROR access to union field is unsafe
                },
        } => {}
    }
    match foo {
        Foo {
            pizza:
                Pizza {
                    topping: Some { .. }
                    //~^ ERROR access to union field is unsafe
                },
        } => {}
        _ => {}
    }
    match foo {
        Foo { tuple: (_a,) } => {} //~ ERROR access to union field is unsafe
    }
    match foo {
        Foo { tuple_struct: TupleStruct(_a) } => {} //~ ERROR access to union field is unsafe
    }
    match foo {
        Foo { array: [_a, _] } => {} //~ ERROR access to union field is unsafe
    }
    match foo {
        Foo { single_variant_enum: SingleVariant::Single {} } => {} //~ ERROR access to union field is unsafe
    }

    // Known bug (#162213): these matches should also be considered non-exhaustive
    match foo {
        Foo { has_private_field: HAS_PRIVATE_FIELD } => {} //~ ERROR access to union field is unsafe
    }
    match foo {
        Foo { has_non_exhaustive_field_list: HAS_NON_EXHAUSTIVE_FIELD_LIST } => {} //~ ERROR access to union field is unsafe
    }
    match foo {
        Foo { has_private_tuple_field: HAS_PRIVATE_TUPLE_FIELD } => {} //~ ERROR access to union field is unsafe
    }
    match foo {
        Foo { has_non_exhaustive_tuple_field_list: HAS_NON_EXHAUSTIVE_TUPLE_FIELD_LIST } => {} //~ ERROR access to union field is unsafe
    }
    match foo {
        Foo { has_non_exhaustive_tuple_field_list: HAS_NON_EXHAUSTIVE_TUPLE_FIELD_LIST } => {} //~ ERROR access to union field is unsafe
    }
    match foo {
        Foo { nested_const: NESTED_CONST } => {} //~ ERROR access to union field is unsafe
    }

    // binding to a tuple, struct, or array pattern is okay if no fields are read
    match foo {
        Foo { zst: () } => {}
    }
    match foo {
        Foo { zst: (..) } => {}
    }
    match foo {
        Foo { zst: UNIT } => {}
    }
    match foo {
        Foo { tuple: (..) } => {}
    }
    match foo {
        Foo { tuple: (_,) } => {}
    }
    match foo {
        Foo { pizza: Pizza { .. } } => {}
    }
    match foo {
        Foo { pizza: Pizza { topping: _ } } => {}
    }
    match foo {
        Foo { tuple_struct: TupleStruct(_) } => {}
    }
    match foo {
        Foo { array: [..] } => {}
    }
    match foo {
        Foo { array: [_, _] } => {}
    }
    match foo {
        Foo { has_private_field: HasPrivateField { .. } } => {}
    }
    match foo {
        Foo { has_non_exhaustive_field_list: HasNonExhaustiveFieldList { .. } } => {}
    }
    match foo {
        Foo { local_non_exhaustive_field_list: LOCAL_NON_EXHAUSTIVE_FIELD_LIST } => {}
    }
    match foo {
        Foo { mixed_visibility_union: MixedVisibilityUnion { zst: () } } => {}
    }

    // binding to wildcard is okay
    match foo {
        Foo { bar: _ } => {}
    }
    let Foo { bar: _ } = foo;
}
