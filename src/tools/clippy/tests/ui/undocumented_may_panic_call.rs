//@aux-build:macro_rules.rs
#![warn(clippy::undocumented_may_panic_call)]
#![allow(clippy::manual_is_multiple_of, clippy::short_circuit_statement)]

extern crate macro_rules;

#[clippy::may_panic]
pub fn dangerous(n: usize) -> bool {
    if n % 2 == 0 {
        panic!()
    }
    true
}

fn main() {
    let _ = dangerous(1);
    //~^ undocumented_may_panic_call

    // Panic: This is safe, 1 is an odd number
    let _ = dangerous(1);
}

#[rustfmt::skip]
fn deeper() {
    let v: Vec<usize> = vec![];
    !v.is_empty()
        && dangerous(1);
    //~^ undocumented_may_panic_call
    !v.is_empty()
    // Panic: This is safe, 1 is an odd number
        && dangerous(1);
}

struct MyStruct {
    value: usize,
}

impl MyStruct {
    #[clippy::may_panic]
    fn panic_method(&self) -> u32 {
        self.value.ilog2()
    }
}

fn test_struct_methods() {
    let s = MyStruct { value: 42 };

    let _ = s.panic_method();
    //~^ undocumented_may_panic_call

    // Panic: value is non-zero, so we are safe!
    let _ = s.panic_method();
}

trait MyTrait {
    #[clippy::may_panic]
    fn trait_panic_method(&self) -> u32;
}

struct TraitImpl {
    data: usize,
}

impl MyTrait for TraitImpl {
    fn trait_panic_method(&self) -> u32 {
        self.data.ilog2()
    }
}

fn test_trait_methods() {
    let t = TraitImpl { data: 10 };

    let _ = t.trait_panic_method();
    //~^ undocumented_may_panic_call

    // Panic: t.data is non-zero
    // We are safe!
    let _ = t.trait_panic_method();
}

fn test_trait_object(t: &dyn MyTrait) {
    // Should lint: no comment
    let _ = t.trait_panic_method();
    //~^ undocumented_may_panic_call

    // Panic: This is safe, just trust me.
    let _ = t.trait_panic_method();
}

// Test macro handling
macro_rules! call_dangerous {
    () => {
        dangerous(1)
    };
}

macro_rules! call_dangerous_with_panic_comment {
    () => {
        // Panic: This is documented inside the macro
        dangerous(1)
    };
}

fn test_macros() {
    // Should lint: macro callsite has no comment
    let _ = call_dangerous!();
    //~^ undocumented_may_panic_call

    // Panic: This documents the macro callsite
    let _ = call_dangerous!();

    // Should not lint: macro body has the comment
    let _ = call_dangerous_with_panic_comment!();
}

macro_rules! nested_macro {
    () => {
        call_dangerous!()
    };
}

fn test_nested_macros() {
    // Should lint: no comment at any level
    let _ = nested_macro!();
    //~^ undocumented_may_panic_call

    // Panic: This documents the outermost macro callsite
    let _ = nested_macro!();
}

#[allow(clippy::needless_bool, clippy::unnecessary_operation, clippy::no_effect)]
fn weird_exprs(t: &dyn MyTrait) {
    // The Panic comment should be in the second If statement
    if if if true { true } else { false } {
        t.trait_panic_method() == 0
    //~^ undocumented_may_panic_call
    } else {
        false
    } {
        true
    } else {
        false
    };
    if if if true { true } else { false } {
        // PANIC: This is safe!
        t.trait_panic_method() == 0
    } else {
        false
    } {
        true
    } else {
        false
    };
    #[clippy::may_panic]
    fn panic<T>(val: T) -> T {
        val
    }
    macro_rules! mac {
        ($t:tt) => {
            $t
        };
    }
    // Panic comment should go on the line before the panic() function
    async {
        {};
        ({
            (u8::max)(
                // PANIC: Safe!
                panic(1) + mac!(2),
                mac!(2),
            )
        });
    };
}

// Test external macros
use macro_rules::external_macro_may_panic;

#[clippy::may_panic]
fn external_may_panic() {}

// External macro wrapped in local macro
macro_rules! wrap_external {
    () => {
        external_macro_may_panic!(external_may_panic)
    };
}

// External macro wrapped in local macro but documented
macro_rules! wrap_external_with_panic_doc {
    () => {
        // Panic: documenting may_panic here
        external_macro_may_panic!(external_may_panic)
    };
}

fn test_external_macro() {
    // Should lint: external macro callsite has no comment
    external_macro_may_panic!(external_may_panic);
    //~^ undocumented_may_panic_call

    // Panic: This documents the external macro callsite
    external_macro_may_panic!(external_may_panic);

    // External macro wrapped in local - should lint at wrap_external callsite
    wrap_external!();
    //~^ undocumented_may_panic_call

    // Panic: wrapping an external macro
    wrap_external!();

    // No comment needed, call is documented inside the local macro
    wrap_external_with_panic_doc!();
}

// Macro that expands to multiple calls
macro_rules! multi_call {
    () => {{
        dangerous(1);
        dangerous(3);
        dangerous(5)
    }};
}

// Macro with the panic comment between calls (should still lint the first)
macro_rules! comment_between {
    () => {{
        dangerous(1);
        // Panic: only covers the second call
        dangerous(3)
    }};
}

fn test_macro_with_mutiple_may_panic_calls() {
    // Multiple calls in one macro - all should lint pointing to same callsite
    multi_call!();
    //~^ undocumented_may_panic_call
    //~| undocumented_may_panic_call
    //~| undocumented_may_panic_call

    // Panic: Extremely safe!
    multi_call!();

    // Comment between calls in macro - but macro still has a dangerous call, so lint
    comment_between!();
    //~^ undocumented_may_panic_call

    // Panic: We are safe now!
    comment_between!();
}

// Macro taking an expression that contains a may_panic call
macro_rules! wrap_expr {
    ($e:expr) => {
        $e
    };
}

fn test_macro_with_may_panic_arg() {
    // The may_panic call is in the argument, not the macro body
    wrap_expr!(dangerous(1));
    //~^ undocumented_may_panic_call

    // Panic: Very safe!
    wrap_expr!(dangerous(1));
}
