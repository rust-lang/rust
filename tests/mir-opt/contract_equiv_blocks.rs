//@ test-mir-pass: ContractEquivBlocks
//! This pass contracts equivalent blocks
//! For a full description of the pass, see its original commit message and PR

// positive example: In scope_ends the two locals s and t are dropped at the end of the scope. But
// the function scope has 2 exits (the ? and the normal scope end). The drop of t is inside a larger
// block, but the drop of s gets its own block in both cases, leading to two identical blocks. Their
// source info points to the end of the scope, which is the same (the end of the function)

// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

fn returns_result() -> Result<(), ()> {
    Ok(())
}

// EMIT_MIR contract_equiv_blocks.scope_ends.ContractEquivBlocks.diff
//
// CHECK-LABEL: fn scope_ends
// CHECK: drop(_1)
// CHECK-NEXT: }
// CHECK-NOT: drop(_1) -> [return: bb{{[0-9]+}}, unwind{{(:| unreachable)}}
fn scope_ends(s: String, t: String) -> Result<(), ()> {
    if s == "".to_string() {
        returns_result()?;
    }
    Ok(())
}

// positive example: Match arms defined using a macro. This way, they get the same source info and
// can contract if their right-hand side is identical
//
// EMIT_MIR contract_equiv_blocks.match_from_macro.ContractEquivBlocks.diff
//
// CHECK-LABEL: fn match_from_macro
// CHECK: = const 0_u8;
// CHECK-NOT: = const 0_u8;
macro_rules! m {
    ($($p:pat),*) => {
        fn match_from_macro(x: u8) -> u8 {
            match x {
                $($p => 0,)*
            }
        }
    };
}
m!(0, _);

// negative example: Match arms with identical right-hand sides still get different source info
// attached to them, so no contraction
//
// EMIT_MIR contract_equiv_blocks.identical_match_arms.ContractEquivBlocks.diff
//
// CHECK-LABEL: fn identical_match_arms
// CHECK: = const 0_u8;
// CHECK: }
// CHECK: = const 0_u8;
fn identical_match_arms(x: u8) -> u8 {
    match x {
        0 => 0,
        _ => 0,
    }
}

// negative example: If the spans are the same, but the operations differ, there obviously is no
// contraction either
//
// EMIT_MIR contract_equiv_blocks.match_from_macro2.ContractEquivBlocks.diff
//
// CHECK-LABEL: fn match_from_macro2
// CHECK: _0 = const <u8 as T>::V;
// CHECK: }
// CHECK: _0 = const <() as T>::V;
trait T {
    const V: u8;
}
impl T for () {
    const V: u8 = 0;
}
impl T for u8 {
    const V: u8 = 1;
}

macro_rules! m2 {
    ($($p:pat => $t:ty),*) => {
        fn match_from_macro2(x: u8) -> u8 {
            match x {
                $($p => <$t as T>::V,)*
            }
        }
    };
}
m2!(0 => (), _ => u8);

// negative example: Call is not an allowed terminator, so these should not contract in this case
//
// EMIT_MIR contract_equiv_blocks.match_from_macro3.ContractEquivBlocks.diff
//
// CHECK-LABEL: fn match_from_macro3
// CHECK: _{{[0-9]+}} = returns_result()
// CHECK: }
// CHECK: _{{[0-9]+}} = returns_result()
macro_rules! m3 {
    ($($p:pat),*) => {
        fn match_from_macro3(x: u8) -> u8 {
            let _ = match x {
                $($p => returns_result(),)*
            };
            0
        }
    };
}
m3!(0, _);
