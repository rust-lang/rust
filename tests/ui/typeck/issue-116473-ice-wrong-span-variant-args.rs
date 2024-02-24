// Regression test for ICE #116473.
// The ICE occurs when arguments are specified on an enum variant
// (which is illegal) and the variant and its preceding path are
// located at different places such as in different macros or
// different expansions of the same macro (i.e. when the macro
// calls itself recursively)

enum Enum<T1, T2> {  VariantA { _v1: T1, _v2: T2 }, VariantB }

type EnumUnit = Enum<(), ()>;

// Recursive macro call using a tt metavariable for variant
macro_rules! recursive_tt {
    () => (recursive_tt!(VariantB));
    ($variant:tt) => (if let EnumUnit::$variant::<i32, u32> {} = 5 { true } else { false });
    //~^ ERROR type arguments are not allowed on this type
    //~| ERROR mismatched types
}


// Recursive macro call using an ident metavariable for variant
// (the behaviour is different for tt and ident)
macro_rules! recursive_ident {
    () => (recursive_ident!(VariantB));
    ($variant:ident) => (if let EnumUnit::$variant::<i32, u32> {} = 5 { true } else { false });
    //~^ ERROR type arguments are not allowed on this type
    //~| ERROR mismatched types
}


// Mested macro calls (i.e. one calling another) using a tt
// metavariable for variant
macro_rules! nested1_tt {
    () => (nested2_tt!(VariantB));
}

macro_rules! nested2_tt {
    ($variant:tt) => (if let EnumUnit::$variant::<i32, u32> {} = 5 { true } else { false });
    //~^ ERROR type arguments are not allowed on this type
    //~| ERROR mismatched types
}


// Mested macro calls using an ident metavariable for variant
// (the behaviour is different for tt and ident)
macro_rules! nested1_ident {
    () => (nested2_ident!(VariantB));
}

macro_rules! nested2_ident {
    ($variant:ident) => (if let EnumUnit::$variant::<i32, u32> {} = 5 { true } else { false });
    //~^ ERROR type arguments are not allowed on this type
    //~| ERROR mismatched types
}


// Mested macro calls when args are passed as metavariable
// instead of the enum variant
macro_rules! nested1_tt_args_in_first_macro {
    () => (nested2_tt_args_in_first_macro!(i32, u32));
    //~^ ERROR type arguments are not allowed on this type
}

macro_rules! nested2_tt_args_in_first_macro {
    ($arg1:tt, $arg2:tt) => (if let EnumUnit::VariantB::<$arg1, $arg2> {}
    //~^ ERROR mismatched types
            = 5 { true } else { false });
}

// Mested macro calls when args are passed as metavariable
// instead of the enum variant
macro_rules! nested1_ident_args_in_first_macro {
    () => (nested2_ident_args_in_first_macro!(i32, u32));
}

macro_rules! nested2_ident_args_in_first_macro {
    ($arg1:ident, $arg2:ident) => (if let EnumUnit::VariantB::<$arg1, $arg2> {}
    //~^ ERROR type arguments are not allowed on this type
    //~| ERROR mismatched types
        = 5 { true } else { false });
}

fn main() {
    // Macro cases
    recursive_tt!();
    recursive_ident!();
    nested1_tt!();
    nested1_ident!();
    nested1_tt_args_in_first_macro!();
    nested1_ident_args_in_first_macro!();

    // Regular, non-macro case
    if let EnumUnit::VariantB::<i32, u32> {} = 5 { true } else { false };
    //~^ ERROR type arguments are not allowed on this type
    //~| ERROR mismatched types
}
