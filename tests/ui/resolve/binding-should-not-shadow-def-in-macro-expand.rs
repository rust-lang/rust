//@ check-pass
//@ edition:2018


type FnF = i8;
type BindingF = i16;

fn main() {}

fn f_without_definition_f(f: impl Fn() -> BindingF) {
    let a: BindingF = m!();
    macro_rules! m {() => ( f() )}
    use m;
}

fn fn0(f: impl Fn() -> BindingF) {
    // param f -> fn f -> macro m

    let a0: FnF = m!();
    fn f() -> FnF { 42 }
    let a1: FnF = m!();
    macro_rules! m {() => ( f() )}
    use m;
    let a2: FnF = m!();
}

fn fn1(f: impl Fn() -> BindingF) {
    // param f -> macro m -> fn f

    let a0: FnF = m!();
    macro_rules! m {() => ( f() )}
    use m;
    let a1: FnF = m!();
    fn f() -> FnF { 42 }
    let a2: FnF = m!();
}

fn closure() {
    let c_without_definition_f = |f: Binding| -> {
        let a1: Binding = m!();
        macro_rules! m {() => ( f )}
        use m;
    };

    let c0 = |f: BindingF| {
        // param f -> fn f -> macro m
        let a0: FnF = m!();
        fn f() -> FnF { 42 }
        let a1: FnF = m!();
        macro_rules! m {() => ( f() )}
        use m;
        let a2: FnF = m!();
    };

    let c1 = |f: BindingF| {
        // param f -> macro m -> fn f
        let a0: FnF = m!();
        macro_rules! m {() => ( f() )}
        use m;
        let a1: FnF = m!();
        fn f() -> FnF { 42 }
        let a2: FnF = m!();
    };
}


fn for_loop() {
    // for f -> macro m -> fn f
    for f in 0..42 as BindingF {
        let a0: FnF = m!();
        macro_rules! m {() => ( f() )}
        use m;
        let a1: FnF = m!();
        fn f() -> FnF { 42 }
        let a2: FnF = m!();
    }

    // for f -> fn f -> macro m
    for f in 0..42 as BindingF {
        let a0: FnF = m!();
        fn f() -> FnF { 42 }
        let a1: FnF = m!();
        macro_rules! m {() => ( f() )}
        use m;
        let a2: FnF = m!();
    }
}


fn match_arm() {
    // match f -> macro m -> fn f
    match 42 as BindingF {
        f => {
            let a0: FnF = m!();
            macro_rules! m {() => ( f() )}
            use m;
            let a1: FnF = m!();
            fn f() -> FnF { 42 }
            let a2: FnF = m!();
        }
    }

    // match f -> fn f -> macro m
    match 42 as BindingF {
        f => {
            let a0: FnF = m!();
            fn f() -> FnF { 42 }
            let a1: FnF = m!();
            macro_rules! m {() => ( f() )}
            use m;
            let a2: FnF = m!();
        }
    }
}
