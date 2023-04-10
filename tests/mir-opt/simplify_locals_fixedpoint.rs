// ignore-wasm32 compiled with panic=abort by default
// compile-flags: -Zmir-opt-level=1

fn foo<T>() {
    if let (Some(a), None) = (Option::<u8>::None, Option::<T>::None) {
        if a > 42u8 {

        }
    }
}

fn main() {
    foo::<()>();
}

// EMIT_MIR simplify_locals_fixedpoint.foo.SimplifyLocals-final.diff
