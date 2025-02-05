//@[classic2021] edition: 2024
//@[structural2021] edition: 2024
//@[classic2024] edition: 2021
//@[structural2024] edition: 2021
//! This contains macros in an edition *different* to the one used in `../mixed-editions.rs`, in
//! order to test typing mixed-edition patterns.

#[macro_export]
macro_rules! match_ctor {
    ($p:pat) => {
        [$p]
    };
}

#[macro_export]
macro_rules! match_ref {
    ($p:pat) => {
        &$p
    };
}

#[macro_export]
macro_rules! bind {
    ($i:ident) => {
        $i
    }
}

#[macro_export]
macro_rules! bind_ref {
    ($i:ident) => {
        ref $i
    }
}

#[macro_export]
macro_rules! bind_mut {
    ($i:ident) => {
        mut $i
    }
}

#[macro_export]
macro_rules! bind_ref_mut {
    ($i:ident) => {
        ref mut $i
    }
}
