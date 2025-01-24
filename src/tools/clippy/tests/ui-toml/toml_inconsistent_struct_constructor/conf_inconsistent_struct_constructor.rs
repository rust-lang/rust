#![warn(clippy::inconsistent_struct_constructor)]
#![allow(clippy::redundant_field_names)]
#![allow(clippy::unnecessary_operation)]
#![allow(clippy::no_effect)]

#[derive(Default)]
struct Foo {
    x: i32,
    y: i32,
    z: i32,
}

fn main() {
    let x = 1;
    let y = 1;
    let z = 1;

    Foo { y, x, z: z };

    Foo {
        z: z,
        x,
        ..Default::default()
    };
}

// https://github.com/rust-lang/rust-clippy/pull/13737#discussion_r1859261645
mod field_attributes {
    struct HirId;
    struct BodyVisitor {
        macro_unsafe_blocks: Vec<HirId>,
        expn_depth: u32,
    }
    fn check_body(condition: bool) {
        BodyVisitor {
            #[expect(clippy::bool_to_int_with_if)] // obfuscates the meaning
            expn_depth: if condition { 1 } else { 0 },
            macro_unsafe_blocks: Vec::new(),
        };
    }
}

// https://github.com/rust-lang/rust-clippy/pull/13737#discussion_r1874539800
mod cfgs_between_fields {
    #[allow(clippy::non_minimal_cfg)]
    fn cfg_all() {
        struct S {
            a: i32,
            b: i32,
            #[cfg(all())]
            c: i32,
            d: i32,
        }
        let s = S {
            d: 0,
            #[cfg(all())]
            c: 1,
            b: 2,
            a: 3,
        };
    }

    fn cfg_any() {
        struct S {
            a: i32,
            b: i32,
            #[cfg(any())]
            c: i32,
            d: i32,
        }
        let s = S {
            d: 0,
            #[cfg(any())]
            c: 1,
            b: 2,
            a: 3,
        };
    }
}
