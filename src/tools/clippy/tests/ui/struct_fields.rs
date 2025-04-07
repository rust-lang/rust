//@aux-build:proc_macros.rs

#![warn(clippy::struct_field_names)]
#![allow(unused)]

#[macro_use]
extern crate proc_macros;

struct Data1 {
    field_data1: u8,
    //~^ ERROR: field name ends with the struct's name
    another: u8,
    foo: u8,
    bar: u8,
}

struct Data2 {
    another: u8,
    foo: u8,
    data2_field: u8,
    //~^ ERROR: field name starts with the struct's name
    bar: u8,
}

struct StructData {
    //~^ ERROR: all fields have the same postfix: `data`
    movable_data: u8,
    fixed_data: u8,
    invisible_data: u8,
}

struct DataStruct {
    //~^ ERROR: all fields have the same prefix: `data`
    data_movable: u8,
    data_fixed: u8,
    data_invisible: u8,
}

struct DoublePrefix {
    //~^ ERROR: all fields have the same prefix: `some_data`
    some_data_a: bool,
    some_data_b: i8,
    some_data_c: bool,
}

struct DoublePostfix {
    //~^ ERROR: all fields have the same postfix: `some_data`
    a_some_data: bool,
    b_some_data: i8,
    c_some_data: bool,
}

#[allow(non_snake_case)]
struct NotSnakeCase {
    //~^ ERROR: all fields have the same postfix: `someData`
    a_someData: bool,
    b_someData: i8,
    c_someData: bool,
}
#[allow(non_snake_case)]
struct NotSnakeCase2 {
    //~^ ERROR: all fields have the same prefix: `someData`
    someData_c: bool,
    someData_b: i8,
    someData_a_b: bool,
}

// no error, threshold is 3 fields by default
struct Fooo {
    foo: u8,
    bar: u8,
}

struct NonCaps {
    //~^ ERROR: all fields have the same prefix: `prefix`
    prefix_的: u8,
    prefix_tea: u8,
    prefix_cake: u8,
}

// should not lint
#[allow(clippy::struct_field_names)]
pub mod allowed {
    pub struct PubAllowed {
        some_this: u8,
        some_that: u8,
        some_other_what: u8,
    }
}

// should not lint
struct SomeData {
    foo: u8,
    bar: bool,
    path: u8,
    answer: u8,
}

// should not lint
pub struct NetworkLayer {
    layer1: Vec<u8>,
    layer2: Vec<u8>,
    layer3: Vec<u8>,
    layer4: Vec<u8>,
}

//should not lint
struct North {
    normal: u8,
    no_left: u8,
    no_right: u8,
}

mod issue8324_from_enum_variant_names {
    // 8324: enum_variant_names warns even if removing the suffix would leave an empty string
    struct Phase {
        pre_lookup: u8,
        lookup: u8,
        post_lookup: u8,
    }
}

mod issue9018_from_enum_variant_names {
    struct DoLint {
        //~^ ERROR: all fields have the same prefix: `_type`
        _type_create: u8,
        _type_read: u8,
        _type_update: u8,
        _type_destroy: u8,
    }

    struct DoLint2 {
        //~^ ERROR: all fields have the same prefix: `__type`
        __type_create: u8,
        __type_read: u8,
        __type_update: u8,
        __type_destroy: u8,
    }

    struct DoLint3 {
        //~^ ERROR: all fields have the same prefix: `___type`
        ___type_create: u8,
        ___type_read: u8,
        ___type_update: u8,
        ___type_destroy: u8,
    }

    struct DoLint4 {
        //~^ ERROR: all fields have the same postfix: `_`
        create_: u8,
        read_: u8,
        update_: u8,
        destroy_: u8,
    }

    struct DoLint5 {
        //~^ ERROR: all fields have the same postfix: `__`
        create__: u8,
        read__: u8,
        update__: u8,
        destroy__: u8,
    }

    struct DoLint6 {
        //~^ ERROR: all fields have the same postfix: `___`
        create___: u8,
        read___: u8,
        update___: u8,
        destroy___: u8,
    }

    struct DoLintToo {
        //~^ ERROR: all fields have the same postfix: `type`
        _create_type: u8,
        _update_type: u8,
        _delete_type: u8,
    }

    struct DoNotLint {
        _foo: u8,
        _bar: u8,
        _baz: u8,
    }

    struct DoNotLint2 {
        __foo: u8,
        __bar: u8,
        __baz: u8,
    }
}

mod allow_attributes_on_fields {
    struct Struct {
        #[allow(clippy::struct_field_names)]
        struct_starts_with: u8,
        #[allow(clippy::struct_field_names)]
        ends_with_struct: u8,
        foo: u8,
    }
}

// food field should not lint
struct Foo {
    food: i32,
    a: i32,
    b: i32,
}

struct Proxy {
    proxy: i32,
    //~^ ERROR: field name starts with the struct's name
    unrelated1: bool,
    unrelated2: bool,
}

// should not lint
pub struct RegexT {
    __buffer: i32,
    __allocated: i32,
    __used: i32,
}

mod macro_tests {
    macro_rules! mk_struct {
        () => {
            struct MacroStruct {
                //~^ ERROR: all fields have the same prefix: `some`
                some_a: i32,
                some_b: i32,
                some_c: i32,
            }
        };
    }
    mk_struct!();

    macro_rules! mk_struct2 {
        () => {
            struct Macrobaz {
                macrobaz_a: i32,
                //~^ ERROR: field name starts with the struct's name
                some_b: i32,
                some_c: i32,
            }
        };
    }
    mk_struct2!();

    macro_rules! mk_struct_with_names {
        ($struct_name:ident, $field:ident) => {
            struct $struct_name {
                $field: i32,
                //~^ ERROR: field name starts with the struct's name
                other_something: i32,
                other_field: i32,
            }
        };
    }
    // expands to `struct Foo { foo: i32, ... }`
    mk_struct_with_names!(Foo, foo);

    // expands to a struct with all fields starting with `other` but should not
    // be linted because some fields come from the macro definition and the other from the input
    mk_struct_with_names!(Some, other_data);

    // should not lint when names come from different places
    macro_rules! mk_struct_with_field_name {
        ($field_name:ident) => {
            struct Baz {
                one: i32,
                two: i32,
                $field_name: i32,
            }
        };
    }
    mk_struct_with_field_name!(baz_three);

    // should not lint when names come from different places
    macro_rules! mk_struct_with_field_name {
        ($field_name:ident) => {
            struct Bazilisk {
                baz_one: i32,
                baz_two: i32,
                $field_name: i32,
            }
        };
    }
    mk_struct_with_field_name!(baz_three);

    macro_rules! mk_struct_full_def {
        ($struct_name:ident, $field1:ident, $field2:ident, $field3:ident) => {
            struct $struct_name {
                //~^ ERROR: all fields have the same prefix: `some`
                $field1: i32,
                $field2: i32,
                $field3: i32,
            }
        };
    }
    mk_struct_full_def!(PrefixData, some_data, some_meta, some_other);
}

// should not lint on external code
external! {
    struct DataExternal {
        field_data1: u8,
        another: u8,
        foo: u8,
        bar: u8,
    }

    struct NotSnakeCaseExternal {
        someData_c: bool,
        someData_b: bool,
        someData_a_b: bool,
    }

    struct DoublePrefixExternal {
        some_data_a: bool,
        some_data_b: bool,
        some_data_c: bool,
    }

    struct StructDataExternal {
        movable_data: u8,
        fixed_data: u8,
        invisible_data: u8,
    }

}

// Should not warn
struct Config {
    use_foo: bool,
    use_bar: bool,
    use_baz: bool,
}

struct Use {
    use_foo: bool,
    //~^ ERROR: field name starts with the struct's name
    use_bar: bool,
    //~^ struct_field_names
    use_baz: bool,
    //~^ struct_field_names
}

// should lint on private fields of public structs (renaming them is not breaking-exported-api)
pub struct PubStructFieldNamedAfterStruct {
    pub_struct_field_named_after_struct: bool,
    //~^ ERROR: field name starts with the struct's name
    other1: bool,
    other2: bool,
}
pub struct PubStructFieldPrefix {
    //~^ ERROR: all fields have the same prefix: `field`
    field_foo: u8,
    field_bar: u8,
    field_baz: u8,
}
// ...but should not lint on structs with public fields.
pub struct PubStructPubAndPrivateFields {
    /// One could argue that this field should be linted, but currently, any public field stops all
    /// checking.
    pub_struct_pub_and_private_fields_1: bool,
    pub pub_struct_pub_and_private_fields_2: bool,
}
// nor on common prefixes if one of the involved fields is public
pub struct PubStructPubAndPrivateFieldPrefix {
    pub field_foo: u8,
    field_bar: u8,
    field_baz: u8,
}

fn main() {}
