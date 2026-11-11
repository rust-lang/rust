//@ check-pass
// Ensure non-capturing Closure passes CoerceMany.
fn foo(x: usize) -> usize {
    0
}

fn bar(x: usize) -> usize {
    1
}

fn main() {
    // One FnDef and one non-capturing Closure
    let _ = match 0 {
        0 => foo,
        2 => |a| 2,
        _ => unimplemented!(),
    };

    let _ = match 0 {
        2 => |a| 2,
        0 => foo,
        _ => unimplemented!(),
    };

    let _ = [foo, |a| 2];
    let _ = [|a| 2, foo];



    // Two FnDefs and one non-capturing Closure
    let _ = match 0 {
        0 => foo,
        1 => bar,
        2 => |a| 2,
        _ => unimplemented!(),
    };

    let _ = match 0 {
        0 => foo,
        2 => |a| 2,
        1 => bar,
        _ => unimplemented!(),
    };

    let _ = match 0 {
        2 => |a| 2,
        0 => foo,
        1 => bar,
        _ => unimplemented!(),
    };

    let _ = [foo, bar, |a| 2];
    let _ = [foo, |a| 2, bar];
    let _ = [|a| 2, foo, bar];



    // One FnDef and two non-capturing Closures
    let _ = match 0 {
        0 => foo,
        1 => |a| 1,
        2 => |a| 2,
        _ => unimplemented!(),
    };

    let _ = match 0 {
        1 => |a| 1,
        0 => foo,
        2 => |a| 2,
        _ => unimplemented!(),
    };

    let _ = match 0 {
        1 => |a| 1,
        2 => |a| 2,
        0 => foo,
        _ => unimplemented!(),
    };

    let _ = [foo, |a| 1, |a| 2];
    let _ = [|a| 1, foo, |a| 2];
    let _ = [|a| 1, |a| 2, foo];



    // Three non-capturing Closures
    let _ = match 0 {
        0 => |a: usize| 0,
        1 => |a| 1,
        2 => |a| 2,
        _ => unimplemented!(),
    };

    let _ = [|a: usize| 0, |a| 1, |a| 2];



    // Three non-capturing Closures variable
    let clo0 = |a: usize| 0;
    let clo1 = |a| 1;
    let clo2 = |a| 2;
    let _ = match 0 {
        0 => clo0,
        1 => clo1,
        2 => clo2,
        _ => unimplemented!(),
    };

    let clo0 = |a: usize| 0;
    let clo1 = |a| 1;
    let clo2 = |a| 2;
    let _ = [clo0, clo1, clo2];



    // --- Function pointer related part

    // Closure is not in a variable
    type FnPointer = fn(usize) -> usize;

    let _ = match 0 {
        0 => foo as FnPointer,
        2 => |a| 2,
        _ => unimplemented!(),
    };
    let _ = match 0 {
        2 => |a| 2,
        0 => foo as FnPointer,
        _ => unimplemented!(),
    };
    let _ = [foo as FnPointer, |a| 2];
    let _ = [|a| 2, foo as FnPointer];
    let _ = [foo, bar, |x| x];
    let _ = [foo as FnPointer, bar, |x| x];
    let _ = [foo, bar as FnPointer, |x| x];
    let _ = [foo, bar, (|x| x) as FnPointer];
    let _ = [foo as FnPointer, bar as FnPointer, |x| x];

    // Closure is in a variable
    let x = |a| 2;
    let _ = match 0 {
        0 => foo as FnPointer,
        2 => x,
        _ => unimplemented!(),
    };
    let x = |a| 2;
    let _ = match 0 {
        2 => x,
        0 => foo as FnPointer,
        _ => unimplemented!(),
    };
    let x = |a| 2;
    let _ = [foo as FnPointer, x];
    let _ = [x, foo as FnPointer];

    let x = |a| 2;
    let _ = [foo, bar, x];
    let x: FnPointer = |a| 2;
    let _ = [foo, bar, x];
    let x = |a| 2;
    let _ = [foo, bar as FnPointer, x];
    let x = |a| 2;
    let _ = [foo as FnPointer, bar, x];
    let x = |a| 2;
    let _ = [foo as FnPointer, bar as FnPointer, x];
}
