//@ skip-filecheck
#![crate_type = "lib"]
#![feature(ergonomic_clones)]
#![expect(incomplete_features)]

use std::clone::UseCloned;

#[allow(dead_code)]
#[derive(Clone)]
#[repr(align(2))]
struct Foo(u16);

impl UseCloned for Foo {}

#[repr(packed)]
struct Packed(Foo);

// EMIT_MIR array_repeat.repeat_never.built.after.mir
pub fn repeat_never() -> [Foo; 0] {
    let foo = Foo(42);
    [foo.use; 0]
}

// EMIT_MIR array_repeat.repeat_once.built.after.mir
pub fn repeat_once() -> [Foo; 1] {
    let foo = Foo(42);
    [foo.use; 1]
}

// EMIT_MIR array_repeat.repeat_many.built.after.mir
pub fn repeat_many() -> [Foo; 42] {
    let foo = Foo(42);
    [foo.use; 42]
}

// EMIT_MIR array_repeat.repeat_never_val.built.after.mir
pub fn repeat_never_val() -> [Foo; 0] {
    [Foo(42).use; 0]
}

// EMIT_MIR array_repeat.repeat_once_val.built.after.mir
pub fn repeat_once_val() -> [Foo; 1] {
    [Foo(42).use; 1]
}

// EMIT_MIR array_repeat.repeat_many_val.built.after.mir
pub fn repeat_many_val() -> [Foo; 42] {
    [Foo(42).use; 42]
}

// EMIT_MIR array_repeat.repeat_never_packed.built.after.mir
pub fn repeat_never_packed() -> [Foo; 0] {
    let packed = Packed(Foo(42));
    [packed.0.use; 0]
}

// EMIT_MIR array_repeat.repeat_once_packed.built.after.mir
pub fn repeat_once_packed() -> [Foo; 1] {
    let packed = Packed(Foo(42));
    [packed.0.use; 1]
}

// EMIT_MIR array_repeat.repeat_many_packed.built.after.mir
pub fn repeat_many_packed() -> [Foo; 42] {
    let packed = Packed(Foo(42));
    [packed.0.use; 42]
}

// EMIT_MIR array_repeat.repeat_never_val_packed.built.after.mir
pub fn repeat_never_val_packed() -> [Foo; 0] {
    [Packed(Foo(42)).0.use; 0]
}

// EMIT_MIR array_repeat.repeat_once_val_packed.built.after.mir
pub fn repeat_once_val_packed() -> [Foo; 1] {
    [Packed(Foo(42)).0.use; 1]
}

// EMIT_MIR array_repeat.repeat_many_val_packed.built.after.mir
pub fn repeat_many_val_packed() -> [Foo; 42] {
    [Packed(Foo(42)).0.use; 42]
}
