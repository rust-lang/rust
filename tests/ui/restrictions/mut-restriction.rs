// aux-build: external-mut-restriction.rs

#![feature(mut_restriction)]

extern crate external_mut_restriction as external;

pub mod local {
    #[derive(Default)]
    pub struct TupleStruct(pub mut(self) u8);

    #[derive(Default)]
    pub struct FieldStruct {
        pub mut(self) field: u8,
    }

    #[derive(Default)]
    pub enum Enum {
        #[default]
        Default,
        Tuple(mut(self) u8),
        Field { mut(self) field: u8 },
    }
}

fn mut_ref(
    local_tuple_struct: &mut local::TupleStruct,
    local_field_struct: &mut local::FieldStruct,
    local_enum: &mut local::Enum
) {
    local_tuple_struct.0 = 1; //~ ERROR field cannot be mutated outside `local`
    local_field_struct.field = 1; //~ ERROR field cannot be mutated outside `local`
    match local_enum {
        local::Enum::Default => {}
        local::Enum::Tuple(ref mut a) => {} //~ ERROR field cannot be mutated outside `local`
        local::Enum::Field { ref mut field } => {} //~ ERROR field cannot be mutated outside `local`
    }
}

fn mut_ptr(a: *mut local::TupleStruct, b: *mut local::FieldStruct) {
    // unsafe doesn't matter
    unsafe {
        (*a).0 = 1; //~ ERROR field cannot be mutated outside `local`
        (*b).field = 1; //~ ERROR field cannot be mutated outside `local`
    }
}

fn main() {
    let mut local_tuple_struct = local::TupleStruct::default();
    let mut local_field_struct = local::FieldStruct::default();
    let mut local_enum = local::Enum::default();

    local_tuple_struct.0 = 1; //~ ERROR field cannot be mutated outside `local`
    local_field_struct.field = 1; //~ ERROR field cannot be mutated outside `local`
    match local_enum {
        local::Enum::Default => {}
        local::Enum::Tuple(ref mut a) => {} //~ ERROR field cannot be mutated outside `local`
        local::Enum::Field { ref mut field } => {} //~ ERROR field cannot be mutated outside `local`
    }
    std::ptr::addr_of_mut!(local_tuple_struct.0); //~ ERROR field cannot be mutated outside `local`
    std::ptr::addr_of_mut!(local_field_struct.field); //~ ERROR field cannot be mutated outside `local`

    &mut local_tuple_struct.0; //~ ERROR field cannot be mutated outside `local`
    &mut local_field_struct.field; //~ ERROR field cannot be mutated outside `local`

    let mut closure = || {
        local_tuple_struct.0 = 1; //~ ERROR field cannot be mutated outside `local`
        local_field_struct.field = 1; //~ ERROR field cannot be mutated outside `local`
    };

    // okay: the mutation occurs inside the function
    closure();
    mut_ref(&mut local_tuple_struct, &mut local_field_struct, &mut local_enum);
    mut_ptr(&mut local_tuple_struct as *mut _, &mut local_field_struct as *mut _);

    // undefined behavior, but not a compile error (it is the same as turning &T into &mut T)
    unsafe { *(&local_tuple_struct.0 as *const _ as *mut _) = 1; }
    unsafe { *(&local_field_struct.field as *const _ as *mut _) = 1; }

    // Check that external items have mut restrictions enforced. We are also checking that the
    // name of the internal module is not present in the error message, as it is not relevant to the
    // user.

    let mut external_top_level_struct = external::TopLevelStruct::default();
    external_top_level_struct.field = 1; //~ ERROR field cannot be mutated outside `external_mut_restriction`

    let mut external_top_level_enum = external::TopLevelEnum::default();
    match external_top_level_enum {
        external::TopLevelEnum::Default => {}
        external::TopLevelEnum::A(ref mut a) => {} //~ ERROR field cannot be mutated outside `external_mut_restriction`
        external::TopLevelEnum::B { ref mut field } => {} //~ ERROR field cannot be mutated outside `external_mut_restriction`
    }

    let mut external_inner_struct = external::inner::InnerStruct::default();
    external_inner_struct.field = 1; //~ ERROR field cannot be mutated outside `external_mut_restriction`

    let mut external_inner_enum = external::inner::InnerEnum::default();
    match external_inner_enum {
        external::inner::InnerEnum::Default => {}
        external::inner::InnerEnum::A(ref mut a) => {} //~ ERROR field cannot be mutated outside `external_mut_restriction`
        external::inner::InnerEnum::B { ref mut field } => {} //~ ERROR field cannot be mutated outside `external_mut_restriction`
    }
}
