// run-rustfix
// aux-build:proc_macros.rs

#![feature(lint_reasons)]
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
        // Even though the output type is specified,
        // this unsuffixed literal is linted to reduce heuristics and keep codebase simple.
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

fn main() {}
