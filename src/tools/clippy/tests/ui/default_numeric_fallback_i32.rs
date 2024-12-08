//@aux-build:proc_macros.rs

#![warn(clippy::default_numeric_fallback)]
#![allow(
    unused,
    clippy::never_loop,
    clippy::no_effect,
    clippy::unnecessary_operation,
    clippy::branches_sharing_code,
    clippy::let_unit_value,
    clippy::let_with_type_underscore
)]

extern crate proc_macros;
use proc_macros::{external, inline_macros};

mod basic_expr {
    fn test() {
        // Should lint unsuffixed literals typed `i32`.
        let x = 22;
        let x = [1, 2, 3];
        let x = if true { (1, 2) } else { (3, 4) };
        let x = match 1 {
            1 => 1,
            _ => 2,
        };

        // Should NOT lint suffixed literals.
        let x = 22_i32;

        // Should NOT lint literals in init expr if `Local` has a type annotation.
        let x: [i32; 3] = [1, 2, 3];
        let x: (i32, i32) = if true { (1, 2) } else { (3, 4) };
        let x: _ = 1;
        let x: u64 = 1;
        const CONST_X: i8 = 1;
    }
}

mod nested_local {
    fn test() {
        let x: _ = {
            // Should lint this because this literal is not bound to any types.
            let y = 1;

            // Should NOT lint this because this literal is bound to `_` of outer `Local`.
            1
        };

        let x: _ = if true {
            // Should lint this because this literal is not bound to any types.
            let y = 1;

            // Should NOT lint this because this literal is bound to `_` of outer `Local`.
            1
        } else {
            // Should lint this because this literal is not bound to any types.
            let y = 1;

            // Should NOT lint this because this literal is bound to `_` of outer `Local`.
            2
        };

        const CONST_X: i32 = {
            // Should lint this because this literal is not bound to any types.
            let y = 1;

            // Should NOT lint this because this literal is bound to `_` of outer `Local`.
            1
        };
    }
}

mod function_def {
    fn ret_i32() -> i32 {
        1
    }

    fn test() {
        // Should lint this because return type is inferred to `i32` and NOT bound to a concrete
        // type.
        let f = || -> _ { 1 };

        // Even though the output type is specified,
        // this unsuffixed literal is linted to reduce heuristics and keep codebase simple.
        let f = || -> i32 { 1 };
    }
}

mod function_calls {
    fn concrete_arg(x: i32) {}

    fn generic_arg<T>(t: T) {}

    fn test() {
        // Should NOT lint this because the argument type is bound to a concrete type.
        concrete_arg(1);

        // Should lint this because the argument type is inferred to `i32` and NOT bound to a concrete type.
        generic_arg(1);

        // Should lint this because the argument type is inferred to `i32` and NOT bound to a concrete type.
        let x: _ = generic_arg(1);
    }
}

mod struct_ctor {
    struct ConcreteStruct {
        x: i32,
    }

    struct GenericStruct<T> {
        x: T,
    }

    fn test() {
        // Should NOT lint this because the field type is bound to a concrete type.
        ConcreteStruct { x: 1 };

        // Should lint this because the field type is inferred to `i32` and NOT bound to a concrete type.
        GenericStruct { x: 1 };

        // Should lint this because the field type is inferred to `i32` and NOT bound to a concrete type.
        let _ = GenericStruct { x: 1 };
    }
}

mod enum_ctor {
    enum ConcreteEnum {
        X(i32),
    }

    enum GenericEnum<T> {
        X(T),
    }

    fn test() {
        // Should NOT lint this because the field type is bound to a concrete type.
        ConcreteEnum::X(1);

        // Should lint this because the field type is inferred to `i32` and NOT bound to a concrete type.
        GenericEnum::X(1);
    }
}

mod method_calls {
    struct StructForMethodCallTest;

    impl StructForMethodCallTest {
        fn concrete_arg(&self, x: i32) {}

        fn generic_arg<T>(&self, t: T) {}
    }

    fn test() {
        let s = StructForMethodCallTest {};

        // Should NOT lint this because the argument type is bound to a concrete type.
        s.concrete_arg(1);

        // Should lint this because the argument type is bound to a concrete type.
        s.generic_arg(1);
    }
}

mod in_macro {
    use super::*;

    // Should lint in internal macro.
    #[inline_macros]
    fn internal() {
        inline!(let x = 22;);
    }

    // Should NOT lint in external macro.
    fn external() {
        external!(let x = 22;);
    }
}

fn check_expect_suppression() {
    #[expect(clippy::default_numeric_fallback)]
    let x = 21;
}

mod type_already_inferred {
    // Should NOT lint if bound to return type
    fn ret_i32() -> i32 {
        1
    }

    // Should NOT lint if bound to return type
    fn ret_if_i32(b: bool) -> i32 {
        if b { 100 } else { 0 }
    }

    // Should NOT lint if bound to return type
    fn ret_i32_tuple() -> (i32, i32) {
        (0, 1)
    }

    // Should NOT lint if bound to return type
    fn ret_stmt(b: bool) -> (i32, i32) {
        if b {
            return (0, 1);
        }
        (0, 0)
    }

    #[allow(clippy::useless_vec)]
    fn vec_macro() {
        // Should NOT lint in `vec!` call if the type was already stated
        let data_i32: Vec<i32> = vec![1, 2, 3];
        let data_i32 = vec![1, 2, 3];
    }
}

mod issue12159 {
    #![allow(non_upper_case_globals, clippy::exhaustive_structs)]
    pub struct Foo;

    static F: i32 = 1;
    impl Foo {
        const LIFE_u8: u8 = 42;
        const LIFE_i8: i8 = 42;
        const LIFE_u16: u16 = 42;
        const LIFE_i16: i16 = 42;
        const LIFE_u32: u32 = 42;
        const LIFE_i32: i32 = 42;
        const LIFE_u64: u64 = 42;
        const LIFE_i64: i64 = 42;
        const LIFE_u128: u128 = 42;
        const LIFE_i128: i128 = 42;
        const LIFE_usize: usize = 42;
        const LIFE_isize: isize = 42;
        const LIFE_f32: f32 = 42.;
        const LIFE_f64: f64 = 42.;

        const fn consts() {
            const LIFE_u8: u8 = 42;
            const LIFE_i8: i8 = 42;
            const LIFE_u16: u16 = 42;
            const LIFE_i16: i16 = 42;
            const LIFE_u32: u32 = 42;
            const LIFE_i32: i32 = 42;
            const LIFE_u64: u64 = 42;
            const LIFE_i64: i64 = 42;
            const LIFE_u128: u128 = 42;
            const LIFE_i128: i128 = 42;
            const LIFE_usize: usize = 42;
            const LIFE_isize: isize = 42;
            const LIFE_f32: f32 = 42.;
            const LIFE_f64: f64 = 42.;
        }
    }
}

fn main() {}
