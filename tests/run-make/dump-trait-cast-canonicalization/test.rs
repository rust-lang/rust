//! Minimal trait-cast program with multiple delayed-Instance
//! call contexts for exercising the
//! `-Z dump-trait-cast-canonicalization` diagnostic flag.
//!
//! Several sibling `core::cast!` calls from different contexts
//! produce directly-sensitive leaf Instances, and their callers
//! become transitively sensitive delayed Instances. This gives
//! `cascade_canonicalize` at least one depth level with multiple
//! Instances — enough to exercise Phase 1 patching and Phase 3
//! emission accounting.

#![feature(trait_cast)]
#![feature(sized_hierarchy)]
#![crate_type = "bin"]

extern crate core;
use core::marker::TraitMetadataTable;

// ---- trait graph ----

trait Root<'a, 'b>: TraitMetadataTable<dyn Root<'a, 'b>> + core::fmt::Debug {
    fn id(&self) -> u32;
}

trait SubX<'a, 'b>: Root<'a, 'b> {
    fn x_val(&self) -> u32;
}

trait SubY<'a, 'b>: Root<'a, 'b> {
    fn y_val(&self) -> u32;
}

// ---- concrete types ----

#[derive(Debug)]
struct TypeA<'a, 'b> {
    x: &'a u32,
    y: &'b u32,
}

#[derive(Debug)]
struct TypeB<'a, 'b> {
    x: &'a u32,
    y: &'b u32,
}

impl<'a, 'b> Root<'a, 'b> for TypeA<'a, 'b> {
    fn id(&self) -> u32 {
        1
    }
}
impl<'a, 'b> Root<'a, 'b> for TypeB<'a, 'b> {
    fn id(&self) -> u32 {
        2
    }
}

impl<'a, 'b> SubX<'a, 'b> for TypeA<'a, 'b> {
    fn x_val(&self) -> u32 {
        *self.x
    }
}
impl<'a, 'b> SubX<'a, 'b> for TypeB<'a, 'b> {
    fn x_val(&self) -> u32 {
        *self.x * 10
    }
}

impl<'a, 'b> SubY<'a, 'b> for TypeA<'a, 'b> {
    fn y_val(&self) -> u32 {
        *self.y
    }
}
impl<'a, 'b> SubY<'a, 'b> for TypeB<'a, 'b> {
    fn y_val(&self) -> u32 {
        *self.y * 10
    }
}

// ---- multiple coercion/cast contexts ----

#[inline(never)]
fn ctx_equal<'a>(x: &'a u32, y: &'a u32) {
    let a = TypeA { x, y };
    let obj: &dyn Root<'_, '_> = &a;
    let sx = core::cast!(in dyn Root<'_, '_>, obj => dyn SubX<'_, '_>).expect("ctx_equal: subx");
    assert_eq!(sx.x_val(), *x);
    let sy = core::cast!(in dyn Root<'_, '_>, obj => dyn SubY<'_, '_>).expect("ctx_equal: suby");
    assert_eq!(sy.y_val(), *y);
}

#[inline(never)]
fn ctx_interior<'a>(x: &'a u32) {
    let local: u32 = 7;
    let b = TypeB { x, y: &local };
    let obj: &dyn Root<'_, '_> = &b;
    let sx = core::cast!(in dyn Root<'_, '_>, obj => dyn SubX<'_, '_>).expect("ctx_interior: subx");
    assert_eq!(sx.x_val(), *x * 10);
    let sy = core::cast!(in dyn Root<'_, '_>, obj => dyn SubY<'_, '_>).expect("ctx_interior: suby");
    assert_eq!(sy.y_val(), 7 * 10);
}

#[inline(never)]
fn ctx_static() {
    static X: u32 = 100;
    static Y: u32 = 200;
    let b = TypeB { x: &X, y: &Y };
    let obj: &dyn Root<'static, 'static> = &b;
    let sx = core::cast!(in dyn Root<'static, 'static>, obj => dyn SubX<'static, 'static>)
        .expect("ctx_static: subx");
    assert_eq!(sx.x_val(), 100 * 10);
    let sy = core::cast!(in dyn Root<'static, 'static>, obj => dyn SubY<'static, 'static>)
        .expect("ctx_static: suby");
    assert_eq!(sy.y_val(), 200 * 10);
}

fn main() {
    let a: u32 = 3;
    let b: u32 = 5;
    ctx_equal(&a, &b);
    ctx_interior(&a);
    ctx_static();
}
