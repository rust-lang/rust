// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

// Test that we generate StorageDead on unwind paths for coroutines.
//
// Basic block and local names can safely change, but the StorageDead statements
// should not go away.

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

struct Foo(i32);

impl Drop for Foo {
    fn drop(&mut self) {}
}

struct Bar(i32);

fn take<T>(_x: T) {}

// EMIT_MIR coroutine_storage_dead_unwind.main-{closure#0}.StateTransform.before.mir
fn main() {
    let _gen = #[coroutine]
    || {
        let a = Foo(5);
        let b = Bar(6);
        yield;
        take(a);
        take(b);
    };
}
