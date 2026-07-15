//@ run-pass
//! Test that `PassMode::Cast` exposes the `CastTarget` structure for arguments and returns.
//!
//! When a platform ABI requires an aggregate to be passed in registers, rustc represents
//! this as `PassMode::Cast` with a `CastTarget` describing the register layout. This test
//! verifies that the public API exposes the register kinds, sizes, and that register
//! exhaustion correctly transitions arguments from `Cast` to `Indirect { on_stack: true }`.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ only-x86_64-unknown-linux-gnu

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;
#[macro_use]
extern crate rustc_public;

use std::convert::TryFrom;
use std::io::Write;
use std::ops::ControlFlow;

use rustc_public::abi::{CallConvention, PassMode, RegKind};
use rustc_public::mir::mono::Instance;
use rustc_public::{CrateDef, ItemKind};

const CRATE_NAME: &str = "input";

fn test_abi_cast() -> ControlFlow<()> {
    let items = rustc_public::all_local_items();

    // Test Cast on argument: a small struct passed in registers.
    let cast_arg_fn = items
        .iter()
        .find(|item| item.kind() == ItemKind::Fn && item.name() == "input::cast_arg")
        .expect("missing cast_arg");

    let instance = Instance::try_from(*cast_arg_fn).unwrap();
    let abi = instance.fn_abi().unwrap();
    assert_eq!(abi.conv, CallConvention::C);
    match &abi.args[0].mode {
        PassMode::Cast { pad_i32, cast } => {
            assert!(!pad_i32);
            assert_eq!(cast.rest.unit.kind, RegKind::Integer);
            assert!(cast.rest.total.bits() > 0);
        }
        other => panic!("Expected PassMode::Cast for struct arg, got: {:?}", other),
    }

    // Test Cast on return: a small struct returned via registers.
    let cast_ret_fn = items
        .iter()
        .find(|item| item.kind() == ItemKind::Fn && item.name() == "input::cast_ret")
        .expect("missing cast_ret");

    let instance = Instance::try_from(*cast_ret_fn).unwrap();
    let abi = instance.fn_abi().unwrap();
    match &abi.ret.mode {
        PassMode::Cast { pad_i32, cast } => {
            assert!(!pad_i32);
            // A 16-byte struct returned via integer registers.
            assert!(
                cast.rest.unit.kind == RegKind::Integer
                    || cast.prefix.iter().any(|r| r.kind == RegKind::Integer),
                "Expected integer registers for return, got: {:?}",
                cast
            );
        }
        other => panic!("Expected PassMode::Cast for struct return, got: {:?}", other),
    }

    // Test Cast with mixed register kinds: struct with int + float fields.
    let cast_mixed_fn = items
        .iter()
        .find(|item| item.kind() == ItemKind::Fn && item.name() == "input::cast_mixed")
        .expect("missing cast_mixed");

    let instance = Instance::try_from(*cast_mixed_fn).unwrap();
    let abi = instance.fn_abi().unwrap();
    match &abi.args[0].mode {
        PassMode::Cast { pad_i32, cast } => {
            assert!(!pad_i32);
            // On x86_64 SysV, a struct { i64, f64 } uses prefix [Int] + rest Sse,
            // or similar split. Just verify we have register info exposed.
            let has_int = cast.prefix.iter().any(|r| r.kind == RegKind::Integer)
                || cast.rest.unit.kind == RegKind::Integer;
            let has_float = cast.prefix.iter().any(|r| r.kind == RegKind::Float)
                || cast.rest.unit.kind == RegKind::Float;
            assert!(
                has_int && has_float,
                "Expected both integer and float registers, got: {:?}",
                cast
            );
        }
        other => panic!("Expected PassMode::Cast for mixed struct arg, got: {:?}", other),
    }

    // Test multiple cast arguments in one function.
    let cast_multi_fn = items
        .iter()
        .find(|item| item.kind() == ItemKind::Fn && item.name() == "input::cast_multi")
        .expect("missing cast_multi");

    let instance = Instance::try_from(*cast_multi_fn).unwrap();
    let abi = instance.fn_abi().unwrap();
    assert_eq!(abi.conv, CallConvention::C);
    assert_eq!(abi.args.len(), 3);
    // First arg: SmallStruct → Cast
    assert!(matches!(&abi.args[0].mode, PassMode::Cast { .. }));
    // Second arg: u64 → Direct (scalar)
    assert!(matches!(&abi.args[1].mode, PassMode::Direct(_)));
    // Third arg: MixedStruct → Cast with both int and float registers
    match &abi.args[2].mode {
        PassMode::Cast { cast, .. } => {
            let has_int = cast.prefix.iter().any(|r| r.kind == RegKind::Integer)
                || cast.rest.unit.kind == RegKind::Integer;
            let has_float = cast.prefix.iter().any(|r| r.kind == RegKind::Float)
                || cast.rest.unit.kind == RegKind::Float;
            assert!(has_int && has_float, "Expected mixed registers, got: {:?}", cast);
        }
        other => panic!("Expected PassMode::Cast for third arg, got: {:?}", other),
    }

    // Test stack spill: same type can have different PassModes when registers are exhausted.
    // On x86_64 SysV, integer args use up to 6 registers (rdi, rsi, rdx, rcx, r8, r9).
    // TwoWords uses 2 registers each, so the 4th one spills to the stack.
    let cast_spill_fn = items
        .iter()
        .find(|item| item.kind() == ItemKind::Fn && item.name() == "input::cast_spill")
        .expect("missing cast_spill");

    let instance = Instance::try_from(*cast_spill_fn).unwrap();
    let abi = instance.fn_abi().unwrap();
    assert_eq!(abi.conv, CallConvention::C);
    assert_eq!(abi.args.len(), 4);
    // First three TwoWords fit in registers (2 regs each = 6 total) → Cast
    for i in 0..3 {
        assert!(
            matches!(&abi.args[i].mode, PassMode::Cast { .. }),
            "Expected arg {} to be Cast, got: {:?}",
            i,
            abi.args[i].mode
        );
    }
    // Fourth TwoWords has no registers left → Indirect (on stack)
    assert!(
        matches!(&abi.args[3].mode, PassMode::Indirect { on_stack: true, .. }),
        "Expected arg 3 to be Indirect on stack, got: {:?}",
        abi.args[3].mode
    );

    ControlFlow::Continue(())
}

fn main() {
    let path = "pass_mode_input.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "-Cpanic=abort".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_abi_cast).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        #[repr(C)]
        pub struct SmallStruct {{
            pub a: u8,
            pub b: u16,
            pub c: u32,
        }}

        #[repr(C)]
        pub struct TwoWords {{
            pub a: u64,
            pub b: u64,
        }}

        #[repr(C)]
        pub struct MixedStruct {{
            pub i: i64,
            pub f: f64,
        }}

        pub extern "C" fn cast_arg(s: SmallStruct) -> u64 {{
            (s.a as u64) + (s.b as u64) + (s.c as u64)
        }}

        pub extern "C" fn cast_ret(x: u64) -> TwoWords {{
            TwoWords {{ a: x, b: x + 1 }}
        }}

        pub extern "C" fn cast_mixed(s: MixedStruct) -> f64 {{
            (s.i as f64) + s.f
        }}

        pub extern "C" fn cast_multi(s: SmallStruct, x: u64, m: MixedStruct) -> u64 {{
            (s.a as u64) + x + (m.i as u64)
        }}

        pub extern "C" fn cast_spill(a: TwoWords, b: TwoWords, c: TwoWords, d: TwoWords) -> u64 {{
            a.a + b.a + c.a + d.a
        }}
    "#
    )?;
    Ok(())
}
