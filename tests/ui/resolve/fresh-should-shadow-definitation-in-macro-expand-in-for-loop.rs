//@ check-pass
//@ edition:2018

// issue#95237

type FnF = i8;
type BindingF = i16;

fn f0() {
    // for f -> macro m -> fn f

    let a0: BindingF = m!();        // TODO: ERROR: defined later
    let end: BindingF = 42;
    for f in 0..end {
        macro_rules! m {() => ( f() )}
        use m;
        fn f() {}
    }
    let a1: BindingF = m!();

    let a2: BindingF = m!();
    fn f() -> FnF { 42 }
    let a3: BindingF = m!();
}

fn f1() {
    // let f -> fn f -> macro m

    let a0: BindingF = m!();        // TODO: ERROR: defined later
    let f = || -> BindingF { 42 };
    let a1: BindingF = m!();
    fn f() -> FnF { 42 }
    let a2: BindingF = m!();
    macro_rules! m {() => ( f() )}
    use m;
    let a3: BindingF = m!();
}

fn f2() {
    // fn f -> let f -> macro m

    let a0: BindingF = m!();         // TODO: ERROR: defined later
    fn f() -> FnF { 42 }
    let a1: BindingF = m!();         // TODO: ERROR: defined later
    let f = || -> BindingF { 42 };
    let a2: BindingF = m!();
    macro_rules! m {() => ( f() )}
    use m;
    let a3: BindingF = m!();
}

fn f3() {
    // fn f -> macro m -> let f

    let a0: FnF = m!();
    fn f() -> FnF { 42 }
    let a1: FnF = m!();
    macro_rules! m {() => ( f() )}
    use m;
    let a2: FnF = m!();
    let f = || -> BindingF { 42 };
    let a3: FnF = m!();
}

fn f4() {
    // macro m -> fn f -> let f;

    let a0: FnF = m!();
    macro_rules! m {() => ( f() )}
    use m;
    let a1: FnF = m!();
    fn f() -> FnF { 42 }
    let a2: FnF = m!();
    let f = || -> BindingF { 42 };
    let a3: FnF = m!();
}

fn f5() {
    // macro m -> let f -> fn f;

    let a0: FnF = m!();
    macro_rules! m {() => ( f() )}
    use m;
    let a1: FnF = m!();
    let f = || -> BindingF { 42 };
    let a2: FnF = m!();
    fn f() -> FnF { 42 }
    let a3: FnF = m!();
}

fn main () {}
