//@ edition:2018

// issue#95237

type FnF = i8;
type BindingF = i16;

fn f_without_definition_f() {
    let f = || -> BindingF { 42 };
    let a: BindingF = m!();
    macro_rules! m {() => ( f() )}
    use m;
}

fn f_without_closure_f0() {
    fn f() -> FnF { 42 }
    let a: FnF = m!();
    macro_rules! m {() => ( f() )}
    use m;
}

fn f0() {
    // let f -> macro m -> fn f

    let a0: BindingF = m!();        //~ NOTE in this expansion of m!
    let f = || -> BindingF { 42 };
    let a1: BindingF = m!();
    macro_rules! m {() => ( f() )}  //~ ERROR cannot find function `f` in this scope
                                    //~| NOTE not found in this scope
    use m;
    let a2: BindingF = m!();
    fn f() -> FnF { 42 }
    let a3: BindingF = m!();
}

fn f1() {
    // let f -> fn f -> macro m

    let a0: BindingF = m!();        //~ NOTE in this expansion of m!
    let f = || -> BindingF { 42 };
    let a1: BindingF = m!();
    fn f() -> FnF { 42 }
    let a2: BindingF = m!();
    macro_rules! m {() => ( f() )}  //~ ERROR cannot find function `f` in this scope
                                    //~| NOTE not found in this scope
    use m;
    let a3: BindingF = m!();
}

fn f2() {
    // fn f -> let f -> macro m

    let a0: BindingF = m!();         //~ NOTE in this expansion of m!
    fn f() -> FnF { 42 }
    let a1: BindingF = m!();         //~ NOTE in this expansion of m!
    let f = || -> BindingF { 42 };
    let a2: BindingF = m!();
    macro_rules! m {() => ( f() )}  //~ ERROR cannot find function `f` in this scope
                                    //~| ERROR cannot find function `f` in this scope
                                    //~| NOTE not found in this scope
                                    //~| NOTE not found in this scope
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

fn tuple_f() {
    // fn f -> let f in tuple -> macro m

    let a0: BindingF = m!();                    //~ NOTE in this expansion of m!
    fn f() -> FnF { 42 }
    let a1: BindingF = m!();                    //~ NOTE in this expansion of m!
    let (f, _) = (|| -> BindingF { 42 }, ());
    let a2: BindingF = m!();
    macro_rules! m {() => ( f() )}              //~ ERROR cannot find function `f` in this scope
                                                //~| ERROR cannot find function `f` in this scope
                                                //~| NOTE not found in this scope
                                                //~| NOTE not found in this scope
    use m;
    let a3: BindingF = m!();
}

fn multiple() {
    fn f() -> FnF { 42 }
    let f = || -> BindingF { 42 };

    let m0_0: BindingF = m0!();
    let m1_0: BindingF = m1!();
    let m2_0: i32 = m2!();              //~ NOTE in this expansion of m2!

    macro_rules! m0 {
        () => { f() }
    }
    macro_rules! m1 {
        () => { f() }
    }

    let m0_1: BindingF = m0!();
    let m1_1: BindingF = m1!();
    let m2_1: i32 = m2!();              //~ NOTE in this expansion of m2!

    let f = || -> i32 { 42 };

    let m0_2: BindingF = m0!();
    let m1_2: BindingF = m1!();
    let m2_2: i32 = m2!();

    macro_rules! m2 {
        () => { f() }                   //~ ERROR cannot find function `f` in this scope
    }                                   //~| ERROR cannot find function `f` in this scope
                                        //~| NOTE not found in this scope
                                        //~| NOTE not found in this scope

    let m0_3: BindingF = m0!();
    let m1_3: BindingF = m1!();
    let m2_3: i32 = m2!();

    use {m0, m1, m2};
}

fn main () {}
