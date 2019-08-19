// run-pass
#![allow(unused_variables)]
fn some_or_wildcard(r: &Option<i32>, b: &i32) {
    let _: &i32 = match r {
        Some(a) => a,
        _ => b,
    };
}

fn none_or_wildcard(r: &Option<i32>, b: &i32) {
    let _: &i32 = match r {
        None => b,
        _ => b,
    };
}

fn some_or_ref_none(r: &Option<i32>, b: &i32) {
    let _: &i32 = match r {
        Some(a) => a,
        &None => b,
    };
}

fn ref_some_or_none(r: &Option<i32>, b: &i32) {
    let _: &i32 = match r {
        &Some(ref a) => a,
        None => b,
    };
}

fn some_or_self(r: &Option<i32>) {
    let _: &Option<i32> = match r {
        Some(n) => {
            let _: &i32 = n;
            r
        },
        x => x,
    };
}

fn multiple_deref(r: &&&&&Option<i32>) {
    let _: i32 = match r {
        Some(a) => *a,
        None => 5,
    };
}

fn match_with_or() {
    // FIXME(tschottdorf): #44912.
    //
    // let x = &Some((3, 3));
    // let _: &i32 = match x {
    //     Some((x, 3)) | &Some((ref x, 5)) => x,
    //     _ => &5i32,
    // };
}

fn nested_mixed() {
    match (&Some(5), &Some(6)) {
        (Some(a), &Some(mut b)) => {
            // Here, the `a` will be `&i32`, because in the first half of the tuple
            // we hit a non-reference pattern and shift into `ref` mode.
            //
            // In the second half of the tuple there's no non-reference pattern,
            // so `b` will be `i32` (bound with `move` mode). Moreover, `b` is
            // mutable.
            let _: &i32 = a;
            b = 7;
            let _: i32 = b;
        },
        _ => {},
    };
}

fn nested_mixed_multiple_deref_1() {
    let x = (1, &Some(5));
    let y = &Some(x);
    match y {
        Some((a, Some(b))) => {
            let _: &i32 = a;
            let _: &i32 = b;
        },
        _ => {},
    };
}

fn nested_mixed_multiple_deref_2() {
    let x = &Some(5);
    let y = &x;
    match y {
        Some(z) => {
            let _: &i32 = z;
        },
        _ => {},
    }
}

fn new_mutable_reference() {
    let mut x = &mut Some(5);
    match &mut x {
        Some(y) => {
            *y = 5;
        },
        None => { },
    }

    match &mut x {
        Some(y) => {
            println!("{}", *y);
        },
        None => {},
    }
}

fn let_implicit_ref_binding() {
    struct Foo(i32);

    // Note that these rules apply to any pattern matching
    // whether it be in a `match` or a `let`.
    // For example, `x` here is a `ref` binding:
    let Foo(x) = &Foo(3);
    let _: &i32 = x;
}

fn explicit_mut_binding() {
    match &Some(5i32) {
        Some(mut n) => {
            n += 1;
            let _ = n;
        }
        None => {},
    };

    match &mut Some(5i32) {
        Some(n) => {
            *n += 1;
            let _ = n;
        }
        None => {},
    };

    match &mut &mut Some(5i32) {
        Some(n) => {
             let _: &mut i32 = n;
        }
        None => {},
    };
}

fn tuple_mut_and_mut_mut() {
    match (Some(5i32), &Some(5i32)) {
        (Some(n), Some(m)) => {
            // `n` and `m` are bound as immutable references. Make new references from them to
            // assert that.
            let r = n;
            let _ = r;
            let q = m;
            let _ = q;

            // Assert the types. Note that we use `n` and `m` here which would fail had they been
            // moved due to the assignments above.
            let _: i32 = n;
            let _: &i32 = m;
        }
        (_, _) => {},
    };

    match (&Some(5i32), &&Some(5i32)) {
        (Some(n), Some(m)) => {
            let _: &i32 = n;
            let _: &i32 = m;
        }
        (_, _) => {},
    };

    match &mut &mut (Some(5i32), Some(5i32)) {
        (Some(n), Some(m)) => {
            // Dereferenced through &mut &mut, so a mutable binding results.
            let _: &mut i32 = n;
            let _: &mut i32 = m;
        }
        (_, _) => {},
    };

    match (&mut Some(5i32), &mut &mut Some(5i32)) {
        (Some(n), Some(m)) => {
            let _: &mut i32 = n;
            let _: &mut i32 = m;
        }
        (_, _) => {},
    };
}

fn min_mir_embedded_type() {
    // The reduced invocation that an ICE was diagnosed with (was consuming
    // adjustments in wrong order).
    match (0u8, &&Some(5i32)) {
        (_, Some(m)) => {
            let _: &i32 = m;
        }
        (_, _) => {},
    };
}

fn no_autoderef() {
    // Binding.
    let x = &3;
    println!("{}", *x);

    // Wildcard.
    let _ = &3;

    // Constant of generic type (string)
    const Y: &'static str = "foo";
    assert_eq!(0, match "foo" {
        Y => 0,
        _ => 1,
    });

    // Reference pattern.
    let &x = &3;
}

pub fn main() {
    let r: &Option<i32> = &Some(3);
    let b = &4i32;

    none_or_wildcard(r, b);
    some_or_wildcard(r, b);
    some_or_ref_none(r, b);
    ref_some_or_none(r, b);

    some_or_self(r);
    multiple_deref(&&&&r);
    match_with_or();

    nested_mixed();
    nested_mixed_multiple_deref_1();
    nested_mixed_multiple_deref_2();

    new_mutable_reference();
    explicit_mut_binding();
    tuple_mut_and_mut_mut();
    min_mir_embedded_type();

    let_implicit_ref_binding();

    no_autoderef();
}
