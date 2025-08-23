//@ edition:2018

// issue#95237

type FnF = i8;
type BindingF = i16;

fn f_without_definition_f0() {
    // let f -> macro m
    let f = || -> BindingF { 42 };
    let a0: BindingF = m!();
    macro_rules! m {() => ( f() )}
    use m;
    let a1: BindingF = m!();
}

fn f_without_definition_f1() {
    // macro m -> let f
    let a: BindingF = m!();         //~ NOTE in this expansion of m!
    macro_rules! m {() => ( f() )}  //~ ERROR cannot find function `f` in this scope
                                    //~| NOTE not found in this scope
    use m;
    let f = || -> BindingF { 42 };
}

fn f_without_closure_f0() {
    // fn f -> macro f
    fn f() -> FnF { 42 }
    let a: FnF = m!();
    macro_rules! m {() => ( f() )}
    use m;
}

fn f_without_closure_f1() {
    // macro f -> fn f
    let a: FnF = m!();
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> FnF { 42 }
}

fn ff0() {
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

fn ff1() {
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

fn ff2() {
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

fn ff3() {
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

fn ff4() {
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

fn ff5() {
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

fn f_with_macro_export0() {
    let a: BindingF = 42;               //~ NOTE `a` is defined here
                                        //~| NOTE `a` is defined here
                                        //~| NOTE `a` is defined here

    #[macro_export]
    macro_rules! m4 { () => { a } }     //~ ERROR cannot access the local binding `a`
                                        //~| ERROR cannot access the local binding `a`
                                        //~| ERROR cannot access the local binding `a`
                                        //~| ERROR cannot find value `a` in this scope
                                        //~| ERROR cannot find value `a` in this scope
                                        //~| NOTE not found in this scope
                                        //~| NOTE not found in this scope
    let b: BindingF = crate::m4!();
}

fn f_use_macro_export_1() {
    fn a() {}

    crate::m4!();                   //~ NOTE in this expansion of crate::m4!
    crate::m5!();                   //~ NOTE in this expansion of crate::m5!
}

fn f_use_macro_export_2() {
    crate::m4!();                   //~ NOTE in this expansion of crate::m4!
                                    //~| NOTE in this expansion of crate::m4!
    crate::m5!();                   //~ NOTE in this expansion of crate::m5!
                                    //~| NOTE in this expansion of crate::m5!
}

fn f_use_macro_export_3() {
    let a = 42;
    crate::m4!();                   //~ NOTE in this expansion of crate::m4!
                                    //~| NOTE in this expansion of crate::m4!
    crate::m5!();                   //~ NOTE in this expansion of crate::m5!
                                    //~| NOTE in this expansion of crate::m5!
}

fn f_with_macro_export1() {
    let a: BindingF = 42;               //~ NOTE `a` is defined here
                                        //~| NOTE `a` is defined here
                                        //~| NOTE `a` is defined here

    #[macro_export]
    macro_rules! m5 { () => { a } }     //~ ERROR cannot access the local binding `a`
                                        //~| ERROR cannot access the local binding `a`
                                        //~| ERROR cannot access the local binding `a`
                                        //~| ERROR cannot find value `a` in this scope
                                        //~| ERROR cannot find value `a` in this scope
                                        //~| NOTE not found in this scope
                                        //~| NOTE not found in this scope

}

fn main() {}
