//@ compile-flags: -Zlint-mir=no
//@ test-mir-pass: ReferencePropagation
//@ needs-unwind

#![feature(core_intrinsics, custom_mir)]

#[inline(never)]
fn opaque(_: impl Sized) {}

fn reference_propagation<'a, T: Copy>(single: &'a T, mut multiple: &'a T) {
    // CHECK-LABEL: fn reference_propagation(

    // Propagation through a reference.
    {
        // CHECK: bb0: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &[[a]];
        // CHECK: [[c:_.*]] = copy [[a]];

        let a = 5_usize;
        let b = &a; // This borrow is only used once.
        let c = *b; // This should be optimized.
        opaque(()); // We use opaque to separate cases into basic blocks in the MIR.
    }

    // Propagation through two references.
    {
        // CHECK: bb1: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[a2:_.*]] = const 7_usize;
        // CHECK: [[b:_.*]] = &[[a]];
        // CHECK: [[btmp:_.*]] = &[[a2]];
        // CHECK: [[b]] = copy [[btmp]];
        // CHECK: [[c:_.*]] = copy (*[[b]]);

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
        // CHECK: bb2: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &[[a]];
        // CHECK: [[d:_.*]] = &[[b]];
        // CHECK: [[c:_.*]] = copy [[a]];

        let a = 5_usize;
        let b = &a;
        let d = &b;
        let c = *b; // `b` is immutably borrowed, we know its value, but do not propagate it
        opaque(d); // prevent `d` from being removed.
    }

    // Propagation through a mutably borrowed reference.
    {
        // CHECK: bb3: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &[[a]];
        // CHECK: [[d:_.*]] = &raw mut [[b]];
        // CHECK: [[c:_.*]] = copy (*[[b]]);

        let a = 5_usize;
        let mut b = &a;
        let d = &raw mut b;
        let c = *b; // `b` is mutably borrowed, we cannot know its value.
        opaque(d); // prevent `d` from being removed.
    }

    // Propagation through an escaping borrow.
    {
        // CHECK: bb4: {
        // CHECK: [[a:_.*]] = const 7_usize;
        // CHECK: [[b:_.*]] = &[[a]];
        // CHECK: [[c:_.*]] = copy [[a]];

        let a = 7_usize;
        let b = &a;
        let c = *b;
        opaque(b); // `b` escapes here, but we can still replace immutable borrow
    }

    // Propagation through a transitively escaping borrow.
    {
        // CHECK: bb5: {
        // CHECK: [[a:_.*]] = const 7_usize;
        // CHECK: [[b1:_.*]] = &[[a]];
        // CHECK: [[c:_.*]] = copy [[a]];
        // CHECK: [[b2:_.*]] = copy [[b1]];
        // CHECK: [[c2:_.*]] = copy [[a]];
        // CHECK: [[b3:_.*]] = copy [[b2]];

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
        // CHECK: bb6: {
        // CHECK-NOT: {{_.*}} = &(*_1);
        // CHECK: [[b:_.*]] = copy (*_1);

        let a = &*single;
        let b = *a; // This should be optimized as `*single`.
        opaque(());
    }

    // Propagation a reborrow of a mutated argument.
    {
        // CHECK: bb7: {
        // CHECK: [[a:_.*]] = &(*_2);
        // CHECK: [[tmp:_.*]] = &(*_1);
        // CHECK: _2 = copy [[tmp]];
        // CHECK: [[b:_.*]] = copy (*[[a]]);

        let a = &*multiple;
        multiple = &*single;
        let b = *a; // This should not be optimized.
        opaque(());
    }

    // Fixed-point propagation through a borrowed reference.
    {
        // CHECK: bb8: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &[[a]];
        // CHECK: [[d:_.*]] = &[[b]];
        // CHECK: [[c:_.*]] = copy [[a]];

        let a = 5_usize;
        let b = &a;
        let d = &b; // first round promotes debuginfo for `d`
        let c = *b; // second round propagates this dereference
        opaque(());
    }

    // Fixed-point propagation through a mutably borrowed reference.
    {
        // CHECK: bb9: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &[[a]];
        // CHECK: [[d:_.*]] = &mut [[b]];
        // FIXME this could be [[a]]
        // CHECK: [[c:_.*]] = copy (*[[b]]);

        let a = 5_usize;
        let mut b = &a;
        let d = &mut b; // first round promotes debuginfo for `d`
        let c = *b; // second round propagates this dereference
        opaque(());
    }
}

fn reference_propagation_mut<'a, T: Copy>(single: &'a mut T, mut multiple: &'a mut T) {
    // CHECK-LABEL: fn reference_propagation_mut(

    // Propagation through a reference.
    {
        // CHECK: bb0: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &mut [[a]];
        // CHECK: [[c:_.*]] = copy [[a]];

        let mut a = 5_usize;
        let b = &mut a; // This borrow is only used once.
        let c = *b; // This should be optimized.
        opaque(());
    }

    // Propagation through two references.
    {
        // CHECK: bb1: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[a2:_.*]] = const 7_usize;
        // CHECK: [[b:_.*]] = &mut [[a]];
        // CHECK: [[btmp:_.*]] = &mut [[a2]];
        // CHECK: [[b]] = copy [[btmp]];
        // CHECK: [[c:_.*]] = copy (*[[b]]);

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
        // CHECK: bb2: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &mut [[a]];
        // CHECK: [[d:_.*]] = &[[b]];
        // CHECK: [[c:_.*]] = copy (*[[b]]);

        let mut a = 5_usize;
        let b = &mut a;
        let d = &b;
        let c = *b; // `b` is immutably borrowed, we know its value, but cannot be removed.
        opaque(d); // prevent `d` from being removed.
    }

    // Propagation through a mutably borrowed reference.
    {
        // CHECK: bb3: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &mut [[a]];
        // CHECK: [[d:_.*]] = &raw mut [[b]];
        // CHECK: [[c:_.*]] = copy (*[[b]]);

        let mut a = 5_usize;
        let mut b = &mut a;
        let d = &raw mut b;
        let c = *b; // `b` is mutably borrowed, we cannot know its value.
        opaque(d); // prevent `d` from being removed.
    }

    // Propagation through an escaping borrow.
    {
        // CHECK: bb4: {
        // CHECK: [[a:_.*]] = const 7_usize;
        // CHECK: [[b:_.*]] = &mut [[a]];
        // CHECK: [[c:_.*]] = copy (*[[b]]);

        let mut a = 7_usize;
        let b = &mut a;
        let c = *b;
        opaque(b); // `b` escapes here, so we can only replace immutable borrow
    }

    // Propagation through a transitively escaping borrow.
    {
        // CHECK: bb5: {
        // CHECK: [[a:_.*]] = const 7_usize;
        // CHECK: [[b1:_.*]] = &mut [[a]];
        // CHECK: [[c:_.*]] = copy (*[[b1]]);
        // CHECK: [[b2:_.*]] = copy [[b1]];
        // CHECK: [[c2:_.*]] = copy (*[[b2]]);
        // CHECK: [[b3:_.*]] = copy [[b2]];

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
        // CHECK: bb6: {
        // CHECK-NOT: {{_.*}} = &(*_1);
        // CHECK: [[b:_.*]] = copy (*_1);

        let a = &mut *single;
        let b = *a; // This should be optimized as `*single`.
        opaque(());
    }

    // Propagation a reborrow of a mutated argument.
    {
        // CHECK: bb7: {
        // CHECK: [[a:_.*]] = &mut (*_2);
        // CHECK: [[tmp:_.*]] = &mut (*_1);
        // CHECK: _2 = copy [[tmp]];
        // CHECK: [[b:_.*]] = copy (*[[a]]);

        let a = &mut *multiple;
        multiple = &mut *single;
        let b = *a; // This should not be optimized.
        opaque(());
    }

    // Fixed-point propagation through a borrowed reference.
    {
        // CHECK: bb8: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &mut [[a]];
        // CHECK: [[d:_.*]] = &[[b]];
        // FIXME this could be [[a]]
        // CHECK: [[c:_.*]] = copy (*[[b]]);

        let mut a = 5_usize;
        let b = &mut a;
        let d = &b; // first round promotes debuginfo for `d`
        let c = *b; // second round propagates this dereference
        opaque(());
    }

    // Fixed-point propagation through a mutably borrowed reference.
    {
        // CHECK: bb9: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &mut [[a]];
        // CHECK: [[d:_.*]] = &mut [[b]];
        // FIXME this could be [[a]]
        // CHECK: [[c:_.*]] = copy (*[[b]]);

        let mut a = 5_usize;
        let mut b = &mut a;
        let d = &mut b; // first round promotes debuginfo for `d`
        let c = *b; // second round propagates this dereference
        opaque(());
    }
}

fn reference_propagation_const_ptr<T: Copy>(single: *const T, mut multiple: *const T) {
    // CHECK-LABEL: fn reference_propagation_const_ptr(

    // Propagation through a reference.
    unsafe {
        // CHECK: bb0: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &raw const [[a]];
        // CHECK: [[c:_.*]] = copy [[a]];

        let a = 5_usize;
        let b = &raw const a; // This borrow is only used once.
        let c = *b; // This should be optimized.
        opaque(());
    }

    // Propagation through two references.
    unsafe {
        // CHECK: bb1: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[a2:_.*]] = const 7_usize;
        // CHECK: [[b:_.*]] = &raw const [[a]];
        // CHECK: [[btmp:_.*]] = &raw const [[a2]];
        // CHECK: [[b]] = copy [[btmp]];
        // CHECK: [[c:_.*]] = copy (*[[b]]);

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
        // CHECK: bb2: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &raw const [[a]];
        // CHECK: [[d:_.*]] = &[[b]];
        // CHECK: [[c:_.*]] = copy [[a]];

        let a = 5_usize;
        let b = &raw const a;
        let d = &b;
        let c = *b; // `b` is immutably borrowed, we know its value, but cannot be removed.
        opaque(d); // prevent `d` from being removed.
    }

    // Propagation through a mutably borrowed reference.
    unsafe {
        // CHECK: bb3: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &raw const [[a]];
        // CHECK: [[d:_.*]] = &raw mut [[b]];
        // CHECK: [[c:_.*]] = copy (*[[b]]);

        let a = 5_usize;
        let mut b = &raw const a;
        let d = &raw mut b;
        let c = *b; // `b` is mutably borrowed, we cannot know its value.
        opaque(d); // prevent `d` from being removed.
    }

    // Propagation through an escaping borrow.
    unsafe {
        // CHECK: bb4: {
        // CHECK: [[a:_.*]] = const 7_usize;
        // CHECK: [[b:_.*]] = &raw const [[a]];
        // CHECK: [[c:_.*]] = copy [[a]];

        let a = 7_usize;
        let b = &raw const a;
        let c = *b;
        opaque(b); // `b` escapes here, so we can only replace immutable borrow
    }

    // Propagation through a transitively escaping borrow.
    unsafe {
        // CHECK: bb5: {
        // CHECK: [[a:_.*]] = const 7_usize;
        // CHECK: [[b1:_.*]] = &raw const [[a]];
        // CHECK: [[c:_.*]] = copy [[a]];
        // CHECK: [[b2:_.*]] = copy [[b1]];
        // CHECK: [[c2:_.*]] = copy [[a]];
        // CHECK: [[b3:_.*]] = copy [[b2]];

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
        // CHECK: bb6: {
        // CHECK-NOT: {{_.*}} = &(*_1);
        // CHECK: [[b:_.*]] = copy (*_1);

        let a = &raw const *single;
        let b = *a; // This should be optimized as `*single`.
        opaque(());
    }

    // Propagation a reborrow of a mutated argument.
    unsafe {
        // CHECK: bb7: {
        // CHECK: [[a:_.*]] = &raw const (*_2);
        // CHECK: [[tmp:_.*]] = &raw const (*_1);
        // CHECK: _2 = copy [[tmp]];
        // CHECK: [[b:_.*]] = copy (*[[a]]);

        let a = &raw const *multiple;
        multiple = &raw const *single;
        let b = *a; // This should not be optimized.
        opaque(());
    }

    // Propagation through a reborrow.
    unsafe {
        // CHECK: bb8: {
        // CHECK: [[a:_.*]] = const 13_usize;
        // CHECK: [[b:_.*]] = &raw const [[a]];
        // CHECK: [[d:_.*]] = &raw const [[a]];
        // CHECK: [[c:_.*]] = copy [[a]];

        let a = 13_usize;
        let b = &raw const a;
        let c = &raw const *b;
        let e = *c;
        opaque(());
    }

    // Fixed-point propagation through a borrowed reference.
    unsafe {
        // CHECK: bb9: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &raw const [[a]];
        // CHECK: [[d:_.*]] = &[[b]];
        // CHECK: [[c:_.*]] = copy [[a]];

        let a = 5_usize;
        let b = &raw const a;
        let d = &b; // first round promotes debuginfo for `d`
        let c = *b; // second round propagates this dereference
        opaque(());
    }

    // Fixed-point propagation through a borrowed reference.
    unsafe {
        // CHECK: bb10: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &raw const [[a]];
        // CHECK: [[d:_.*]] = &mut [[b]];
        // FIXME this could be [[a]]
        // CHECK: [[c:_.*]] = copy (*[[b]]);

        let a = 5_usize;
        let mut b = &raw const a;
        let d = &mut b; // first round promotes debuginfo for `d`
        let c = *b; // second round propagates this dereference
        opaque(());
    }
}

fn reference_propagation_mut_ptr<T: Copy>(single: *mut T, mut multiple: *mut T) {
    // CHECK-LABEL: fn reference_propagation_mut_ptr(

    // Propagation through a reference.
    unsafe {
        // CHECK: bb0: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &raw mut [[a]];
        // CHECK: [[c:_.*]] = copy [[a]];

        let mut a = 5_usize;
        let b = &raw mut a; // This borrow is only used once.
        let c = *b; // This should be optimized.
        opaque(());
    }

    // Propagation through two references.
    unsafe {
        // CHECK: bb1: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[a2:_.*]] = const 7_usize;
        // CHECK: [[b:_.*]] = &raw mut [[a]];
        // CHECK: [[btmp:_.*]] = &raw mut [[a2]];
        // CHECK: [[b]] = copy [[btmp]];
        // CHECK: [[c:_.*]] = copy (*[[b]]);

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
        // CHECK: bb2: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &raw mut [[a]];
        // CHECK: [[d:_.*]] = &[[b]];
        // CHECK: [[c:_.*]] = copy (*[[b]]);

        let mut a = 5_usize;
        let b = &raw mut a;
        let d = &b;
        let c = *b; // `b` is immutably borrowed, we know its value, but cannot be removed.
        opaque(d); // prevent `d` from being removed.
    }

    // Propagation through a mutably borrowed reference.
    unsafe {
        // CHECK: bb3: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &raw mut [[a]];
        // CHECK: [[d:_.*]] = &raw mut [[b]];
        // CHECK: [[c:_.*]] = copy (*[[b]]);

        let mut a = 5_usize;
        let mut b = &raw mut a;
        let d = &raw mut b;
        let c = *b; // `b` is mutably borrowed, we cannot know its value.
        opaque(d); // prevent `d` from being removed.
    }

    // Propagation through an escaping borrow.
    unsafe {
        // CHECK: bb4: {
        // CHECK: [[a:_.*]] = const 7_usize;
        // CHECK: [[b:_.*]] = &raw mut [[a]];
        // CHECK: [[c:_.*]] = copy (*[[b]]);

        let mut a = 7_usize;
        let b = &raw mut a;
        let c = *b;
        opaque(b); // `b` escapes here, so we can only replace immutable borrow
    }

    // Propagation through a transitively escaping borrow.
    unsafe {
        // CHECK: bb5: {
        // CHECK: [[a:_.*]] = const 7_usize;
        // CHECK: [[b1:_.*]] = &raw mut [[a]];
        // CHECK: [[c:_.*]] = copy (*[[b1]]);
        // CHECK: [[b2:_.*]] = copy [[b1]];
        // CHECK: [[c2:_.*]] = copy (*[[b2]]);
        // CHECK: [[b3:_.*]] = copy [[b2]];

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
        // CHECK: bb6: {
        // CHECK-NOT: {{_.*}} = &(*_1);
        // CHECK: [[b:_.*]] = copy (*_1);

        let a = &raw mut *single;
        let b = *a; // This should be optimized as `*single`.
        opaque(());
    }

    // Propagation a reborrow of a mutated argument.
    unsafe {
        // CHECK: bb7: {
        // CHECK: [[a:_.*]] = &raw mut (*_2);
        // CHECK: [[tmp:_.*]] = &raw mut (*_1);
        // CHECK: _2 = copy [[tmp]];
        // CHECK: [[b:_.*]] = copy (*[[a]]);

        let a = &raw mut *multiple;
        multiple = &raw mut *single;
        let b = *a; // This should not be optimized.
        opaque(());
    }

    // Fixed-point propagation through a borrowed reference.
    unsafe {
        // CHECK: bb8: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &raw mut [[a]];
        // CHECK: [[d:_.*]] = &[[b]];
        // FIXME this could be [[a]]
        // CHECK: [[c:_.*]] = copy (*[[b]]);

        let mut a = 5_usize;
        let b = &raw mut a;
        let d = &b; // first round promotes debuginfo for `d`
        let c = *b; // second round propagates this dereference
        opaque(());
    }

    // Fixed-point propagation through a mutably borrowed reference.
    unsafe {
        // CHECK: bb9: {
        // CHECK: [[a:_.*]] = const 5_usize;
        // CHECK: [[b:_.*]] = &raw mut [[a]];
        // CHECK: [[d:_.*]] = &mut [[b]];
        // FIXME this could be [[a]]
        // CHECK: [[c:_.*]] = copy (*[[b]]);

        let mut a = 5_usize;
        let mut b = &raw mut a;
        let d = &mut b; // first round promotes debuginfo for `d`
        let c = *b; // second round propagates this dereference
        opaque(());
    }
}

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn read_through_raw(x: &mut usize) -> usize {
    // CHECK-LABEL: read_through_raw
    // CHECK: bb0: {
    // CHECK-NEXT: _0 = copy (*_1);
    // CHECK-NEXT: _0 = copy (*_1);
    // CHECK-NEXT: return;

    use std::intrinsics::mir::*;
    mir! {
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
    }
}

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn multiple_storage() {
    // CHECK-LABEL: multiple_storage
    // CHECK: _3 = copy (*_2);

    use std::intrinsics::mir::*;
    mir! {
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
            Call(RET = opaque(y), ReturnTo(retblock), UnwindContinue())
        }

        retblock = {
            Return()
        }
    }
}

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn dominate_storage() {
    // CHECK-LABEL: dominate_storage
    // CHECK: _5 = copy (*_2);

    use std::intrinsics::mir::*;
    mir! {
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
            Call(RET = opaque(c), ReturnTo(bb2), UnwindContinue())
        }
        bb2 = {
            StorageDead(x);
            StorageLive(x);
            let d = true;
            match d { false => bb2, _ => bb0 }
        }
    }
}

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn maybe_dead(m: bool) {
    // CHECK-LABEL: fn maybe_dead(
    // CHECK: (*_5) = const 7_i32;

    use std::intrinsics::mir::*;
    mir! {
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
            Call(RET = opaque(u), ReturnTo(bb2), UnwindContinue())
        }
        bb2 = {
            // As `x` may be `StorageDead`, `a` may be dangling, so we do nothing.
            let z = *a;
            Call(RET = opaque(z), ReturnTo(bb3), UnwindContinue())
        }
        bb3 = {
            // As `y` may be `StorageDead`, `b` may be dangling, so we do nothing.
            // This implies that we also do not substitute `b` in `bb0`.
            let t = *b;
            Call(RET = opaque(t), ReturnTo(retblock), UnwindContinue())
        }
        retblock = {
            Return()
        }
    }
}

fn mut_raw_then_mut_shr() -> (i32, i32) {
    // CHECK-LABEL: fn mut_raw_then_mut_shr(
    // CHECK-NOT: (*{{_.*}})

    let mut x = 2;
    let xref = &mut x;
    let xraw = &mut *xref as *mut _;
    let xshr = &*xref;
    // Verify that we completely replace with `x` in both cases.
    let a = *xshr;
    unsafe {
        *xraw = 4;
    }
    (a, x)
}

fn unique_with_copies() {
    // CHECK-LABEL: fn unique_with_copies(
    // CHECK: [[a:_.*]] = const 0_i32;
    // CHECK: [[x:_.*]] = &raw mut [[a]];
    // CHECK-NOT: [[a]]
    // CHECK: [[tmp:_.*]] = copy (*[[x]]);
    // CHECK-NEXT: opaque::<i32>(move [[tmp]])
    // CHECK-NOT: [[a]]
    // CHECK: StorageDead([[a]]);
    // CHECK-NOT: [[a]]
    // CHECK: [[tmp:_.*]] = copy (*[[x]]);
    // CHECK-NEXT: opaque::<i32>(move [[tmp]])

    let y = {
        let mut a = 0;
        let x = &raw mut a;
        // `*y` is not replacable below, so we must not replace `*x`.
        unsafe { opaque(*x) };
        x
    };
    // But rewriting as `*x` is ok.
    unsafe { opaque(*y) };
}

fn debuginfo() {
    // CHECK-LABEL: fn debuginfo(
    // FIXME: This features waits for DWARF implicit pointers in LLVM.
    // CHECK: debug ref_mut_u8 => _{{.*}};
    // CHECK: debug field => _{{.*}};
    // CHECK: debug reborrow => _{{.*}};
    // CHECK: debug variant_field => _{{.*}};
    // CHECK: debug constant_index => _{{.*}};
    // CHECK: debug subslice => _{{.*}};
    // CHECK: debug constant_index_from_end => _{{.*}};
    // CHECK: debug multiple_borrow => _{{.*}};

    struct T(u8);

    let ref_mut_u8 = &mut 5_u8;
    let field = &T(0).0;

    // Verify that we don't emit `&*` in debuginfo.
    let reborrow = &*ref_mut_u8;

    match Some(0) {
        None => {}
        Some(ref variant_field) => {}
    }

    // `constant_index_from_end` and `subslice` should not be promoted, as their value depends
    // on the slice length.
    if let [_, ref constant_index, subslice @ .., ref constant_index_from_end] = &[6; 10][..] {}

    let multiple_borrow = &&&mut T(6).0;
}

fn many_debuginfo() {
    // CHECK-LABEL: fn many_debuginfo(
    // FIXME: This features waits for DWARF implicit pointers in LLVM.
    // CHECK: debug many_borrow => _{{.*}};

    let a = 0;

    // Verify that we do not ICE on deeply nested borrows.
    let many_borrow =
        &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&a;
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
    mut_raw_then_mut_shr();
    unique_with_copies();
    debuginfo();
    many_debuginfo();
}

// EMIT_MIR reference_prop.reference_propagation.ReferencePropagation.diff
// EMIT_MIR reference_prop.reference_propagation_mut.ReferencePropagation.diff
// EMIT_MIR reference_prop.reference_propagation_const_ptr.ReferencePropagation.diff
// EMIT_MIR reference_prop.reference_propagation_mut_ptr.ReferencePropagation.diff
// EMIT_MIR reference_prop.read_through_raw.ReferencePropagation.diff
// EMIT_MIR reference_prop.multiple_storage.ReferencePropagation.diff
// EMIT_MIR reference_prop.dominate_storage.ReferencePropagation.diff
// EMIT_MIR reference_prop.maybe_dead.ReferencePropagation.diff
// EMIT_MIR reference_prop.mut_raw_then_mut_shr.ReferencePropagation.diff
// EMIT_MIR reference_prop.unique_with_copies.ReferencePropagation.diff
// EMIT_MIR reference_prop.debuginfo.ReferencePropagation.diff
