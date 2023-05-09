// unit-test: ReferencePropagation
// needs-unwind

#![feature(raw_ref_op)]
#![feature(core_intrinsics, custom_mir)]

#[inline(never)]
fn opaque(_: impl Sized) {}

fn reference_propagation<'a, T: Copy>(single: &'a T, mut multiple: &'a T) {
    // Propagation through a reference.
    {
        let a = 5_usize;
        let b = &a; // This borrow is only used once.
        let c = *b; // This should be optimized.
        opaque(()); // We use opaque to separate cases into basic blocks in the MIR.
    }

    // Propagation through a two references.
    {
        let a = 5_usize;
        let a2 = 7_usize;
        let mut b = &a;
        b = &a2;
        // `b` is assigned twice, so we cannot propagate it.
        let c = *b;
        opaque(());
    }

    // Propagation through a borrowed reference.
    {
        let a = 5_usize;
        let b = &a;
        let d = &b;
        let c = *b; // `b` is immutably borrowed, we know its value, but do not propagate it
        opaque(());
    }

    // Propagation through a borrowed reference.
    {
        let a = 5_usize;
        let mut b = &a;
        let d = &mut b;
        let c = *b; // `b` is mutably borrowed, we cannot know its value.
        opaque(());
    }

    // Propagation through an escaping borrow.
    {
        let a = 7_usize;
        let b = &a;
        let c = *b;
        opaque(b); // `b` escapes here, but we can still replace immutable borrow
    }

    // Propagation through a transitively escaping borrow.
    {
        let a = 7_usize;
        let b1 = &a;
        let c = *b1;
        let b2 = b1;
        let c2 = *b2;
        let b3 = b2;
        // `b3` escapes here, so we can only replace immutable borrow,
        // for either `b`, `b2` or `b3`.
        opaque(b3);
    }

    // Propagation a reborrow of an argument.
    {
        let a = &*single;
        let b = *a; // This should be optimized as `*single`.
        opaque(());
    }

    // Propagation a reborrow of a mutated argument.
    {
        let a = &*multiple;
        multiple = &*single;
        let b = *a; // This should not be optimized.
        opaque(());
    }
}

fn reference_propagation_mut<'a, T: Copy>(single: &'a mut T, mut multiple: &'a mut T) {
    // Propagation through a reference.
    {
        let mut a = 5_usize;
        let b = &mut a; // This borrow is only used once.
        let c = *b; // This should be optimized.
        opaque(());
    }

    // Propagation through a two references.
    {
        let mut a = 5_usize;
        let mut a2 = 7_usize;
        let mut b = &mut a;
        b = &mut a2;
        // `b` is assigned twice, so we cannot propagate it.
        let c = *b;
        opaque(());
    }

    // Propagation through a borrowed reference.
    {
        let mut a = 5_usize;
        let b = &mut a;
        let d = &b;
        let c = *b; // `b` is immutably borrowed, we know its value, but cannot be removed.
        opaque(());
    }

    // Propagation through a borrowed reference.
    {
        let mut a = 5_usize;
        let mut b = &mut a;
        let d = &mut b;
        let c = *b; // `b` is mutably borrowed, we cannot know its value.
        opaque(());
    }

    // Propagation through an escaping borrow.
    {
        let mut a = 7_usize;
        let b = &mut a;
        let c = *b;
        opaque(b); // `b` escapes here, so we can only replace immutable borrow
    }

    // Propagation through a transitively escaping borrow.
    {
        let mut a = 7_usize;
        let b1 = &mut a;
        let c = *b1;
        let b2 = b1;
        let c2 = *b2;
        let b3 = b2;
        // `b3` escapes here, so we can only replace immutable borrow,
        // for either `b`, `b2` or `b3`.
        opaque(b3);
    }

    // Propagation a reborrow of an argument.
    {
        let a = &mut *single;
        let b = *a; // This should be optimized as `*single`.
        opaque(());
    }

    // Propagation a reborrow of a mutated argument.
    {
        let a = &mut *multiple;
        multiple = &mut *single;
        let b = *a; // This should not be optimized.
        opaque(());
    }
}

fn reference_propagation_const_ptr<T: Copy>(single: *const T, mut multiple: *const T) {
    // Propagation through a reference.
    unsafe {
        let a = 5_usize;
        let b = &raw const a; // This borrow is only used once.
        let c = *b; // This should be optimized.
        opaque(());
    }

    // Propagation through a two references.
    unsafe {
        let a = 5_usize;
        let a2 = 7_usize;
        let mut b = &raw const a;
        b = &raw const a2;
        // `b` is assigned twice, so we cannot propagate it.
        let c = *b;
        opaque(());
    }

    // Propagation through a borrowed reference.
    unsafe {
        let a = 5_usize;
        let b = &raw const a;
        let d = &b;
        let c = *b; // `b` is immutably borrowed, we know its value, but cannot be removed.
        opaque(());
    }

    // Propagation through a borrowed reference.
    unsafe {
        let a = 5_usize;
        let mut b = &raw const a;
        let d = &mut b;
        let c = *b; // `b` is mutably borrowed, we cannot know its value.
        opaque(());
    }

    // Propagation through an escaping borrow.
    unsafe {
        let a = 7_usize;
        let b = &raw const a;
        let c = *b;
        opaque(b); // `b` escapes here, so we can only replace immutable borrow
    }

    // Propagation through a transitively escaping borrow.
    unsafe {
        let a = 7_usize;
        let b1 = &raw const a;
        let c = *b1;
        let b2 = b1;
        let c2 = *b2;
        let b3 = b2;
        // `b3` escapes here, so we can only replace immutable borrow,
        // for either `b`, `b2` or `b3`.
        opaque(b3);
    }

    // Propagation a reborrow of an argument.
    unsafe {
        let a = &raw const *single;
        let b = *a; // This should be optimized as `*single`.
        opaque(());
    }

    // Propagation a reborrow of a mutated argument.
    unsafe {
        let a = &raw const *multiple;
        multiple = &raw const *single;
        let b = *a; // This should not be optimized.
        opaque(());
    }

    // Propagation through a reborrow.
    unsafe {
        let a = 13_usize;
        let b = &raw const a;
        let c = &raw const *b;
        let e = *c;
        opaque(());
    }
}

fn reference_propagation_mut_ptr<T: Copy>(single: *mut T, mut multiple: *mut T) {
    // Propagation through a reference.
    unsafe {
        let mut a = 5_usize;
        let b = &raw mut a; // This borrow is only used once.
        let c = *b; // This should be optimized.
        opaque(());
    }

    // Propagation through a two references.
    unsafe {
        let mut a = 5_usize;
        let mut a2 = 7_usize;
        let mut b = &raw mut a;
        b = &raw mut a2;
        // `b` is assigned twice, so we cannot propagate it.
        let c = *b;
        opaque(());
    }

    // Propagation through a borrowed reference.
    unsafe {
        let mut a = 5_usize;
        let b = &raw mut a;
        let d = &b;
        let c = *b; // `b` is immutably borrowed, we know its value, but cannot be removed.
        opaque(());
    }

    // Propagation through a borrowed reference.
    unsafe {
        let mut a = 5_usize;
        let mut b = &raw mut a;
        let d = &mut b;
        let c = *b; // `b` is mutably borrowed, we cannot know its value.
        opaque(());
    }

    // Propagation through an escaping borrow.
    unsafe {
        let mut a = 7_usize;
        let b = &raw mut a;
        let c = *b;
        opaque(b); // `b` escapes here, so we can only replace immutable borrow
    }

    // Propagation through a transitively escaping borrow.
    unsafe {
        let mut a = 7_usize;
        let b1 = &raw mut a;
        let c = *b1;
        let b2 = b1;
        let c2 = *b2;
        let b3 = b2;
        // `b3` escapes here, so we can only replace immutable borrow,
        // for either `b`, `b2` or `b3`.
        opaque(b3);
    }

    // Propagation a reborrow of an argument.
    unsafe {
        let a = &raw mut *single;
        let b = *a; // This should be optimized as `*single`.
        opaque(());
    }

    // Propagation a reborrow of a mutated argument.
    unsafe {
        let a = &raw mut *multiple;
        multiple = &raw mut *single;
        let b = *a; // This should not be optimized.
        opaque(());
    }
}

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn read_through_raw(x: &mut usize) -> usize {
    use std::intrinsics::mir::*;

    mir!(
        let r1: &mut usize;
        let r2: &mut usize;
        let p1: *mut usize;
        let p2: *mut usize;

        {
            r1 = &mut *x;
            r2 = &mut *r1;
            p1 = &raw mut *r1;
            p2 = &raw mut *r2;

            RET = *p1;
            RET = *p2;
            Return()
        }
    )
}

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn multiple_storage() {
    use std::intrinsics::mir::*;

    mir!(
        let x: i32;
        {
            StorageLive(x);
            x = 5;
            let z = &x;
            StorageDead(x);
            StorageLive(x);
            // As there are multiple `StorageLive` statements for `x`, we cannot know if this `z`'s
            // pointer address is the address of `x`, so do nothing.
            let y = *z;
            Call(RET, retblock, opaque(y))
        }

        retblock = {
            Return()
        }
    )
}

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn dominate_storage() {
    use std::intrinsics::mir::*;

    mir!(
        let x: i32;
        let r: &i32;
        let c: i32;
        let d: bool;
        { Goto(bb0) }
        bb0 = {
            x = 5;
            r = &x;
            Goto(bb1)
        }
        bb1 = {
            let c = *r;
            Call(RET, bb2, opaque(c))
        }
        bb2 = {
            StorageDead(x);
            StorageLive(x);
            let d = true;
            match d { false => bb2, _ => bb0 }
        }
    )
}

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn maybe_dead(m: bool) {
    use std::intrinsics::mir::*;

    mir!(
        let x: i32;
        let y: i32;
        {
            StorageLive(x);
            StorageLive(y);
            x = 5;
            y = 5;
            let a = &x;
            let b = &mut y;
            // As we don't replace `b` in `bb2`, we cannot replace it here either.
            *b = 7;
            // But this can still be replaced.
            let u = *a;
            match m { true => bb1, _ => bb2 }
        }
        bb1 = {
            StorageDead(x);
            StorageDead(y);
            Call(RET, bb2, opaque(u))
        }
        bb2 = {
            // As `x` may be `StorageDead`, `a` may be dangling, so we do nothing.
            let z = *a;
            Call(RET, bb3, opaque(z))
        }
        bb3 = {
            // As `y` may be `StorageDead`, `b` may be dangling, so we do nothing.
            // This implies that we also do not substitute `b` in `bb0`.
            let t = *b;
            Call(RET, retblock, opaque(t))
        }
        retblock = {
            Return()
        }
    )
}

fn main() {
    let mut x = 5_usize;
    let mut y = 7_usize;
    reference_propagation(&x, &y);
    reference_propagation_mut(&mut x, &mut y);
    reference_propagation_const_ptr(&raw const x, &raw const y);
    reference_propagation_mut_ptr(&raw mut x, &raw mut y);
    read_through_raw(&mut x);
    multiple_storage();
    dominate_storage();
    maybe_dead(true);
}

// EMIT_MIR reference_prop.reference_propagation.ReferencePropagation.diff
// EMIT_MIR reference_prop.reference_propagation_mut.ReferencePropagation.diff
// EMIT_MIR reference_prop.reference_propagation_const_ptr.ReferencePropagation.diff
// EMIT_MIR reference_prop.reference_propagation_mut_ptr.ReferencePropagation.diff
// EMIT_MIR reference_prop.read_through_raw.ReferencePropagation.diff
// EMIT_MIR reference_prop.multiple_storage.ReferencePropagation.diff
// EMIT_MIR reference_prop.dominate_storage.ReferencePropagation.diff
// EMIT_MIR reference_prop.maybe_dead.ReferencePropagation.diff
