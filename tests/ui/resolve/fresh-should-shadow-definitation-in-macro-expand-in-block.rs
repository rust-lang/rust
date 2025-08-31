//@ edition:2018

// issue#95237

type FnF = i8;
type BindingF = i16;

fn b_without_definition_f() {
    let f = || -> BindingF { 42 };
    {
        let a: BindingF = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
}

fn b_without_closure_f() {
    fn f() -> FnF { 42 }
    {
        let a: FnF = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
}

fn ff0() {
    // let f -> macro m -> fn f

    {
        let a0: BindingF = m!();        //~ NOTE in this expansion of m!
    }
    let f = || -> BindingF { 42 };      //~ NOTE `f` is defined here
    {
        let a1: BindingF = m!();
    }
    macro_rules! m {() => ( f() )}      //~ ERROR cannot find function `f` in this scope
                                        //~| NOTE not found in this scope
    use m;
    {
        let a2: BindingF = m!();
    }
    fn f() -> FnF { 42 }
    {
        let a3: BindingF = m!();
    }
}

fn ff1() {
    // let f -> fn f -> macro m

    {
        let a0: BindingF = m!();        //~ NOTE in this expansion of m!
    }
    let f = || -> BindingF { 42 };      //~ NOTE `f` is defined here
    {
        let a1: BindingF = m!();
    }
    fn f() -> FnF { 42 }
    {
        let a2: BindingF = m!();
    }
    macro_rules! m {() => ( f() )}      //~ ERROR cannot find function `f` in this scope
                                        //~| NOTE not found in this scope
    use m;
    {
        let a3: BindingF = m!();
    }
}

fn ff2() {
    // fn f -> let f -> macro m

    {
        let a0: BindingF = m!();         //~ NOTE in this expansion of m!
    }
    fn f() -> FnF { 42 }
    {
        let a1: BindingF = m!();         //~ NOTE in this expansion of m!
    }
    let f = || -> BindingF { 42 };       //~ NOTE `f` is defined here
                                         //~| NOTE `f` is defined here
    {
        let a2: BindingF = m!();
    }
    macro_rules! m {() => ( f() )}      //~ ERROR cannot find function `f` in this scope
                                        //~| ERROR cannot find function `f` in this scope
                                        //~| NOTE not found in this scope
                                        //~| NOTE not found in this scope
    use m;
    {
        let a3: BindingF = m!();
    }
}

fn ff3() {
    // fn f -> macro m -> let f

    {
        let a0: FnF = m!();
    }
    fn f() -> FnF { 42 }
    {
        let a1: FnF = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
    {
        let a2: FnF = m!();
    }
    let f = || -> BindingF { 42 };
    {
        let a3: FnF = m!();
    }
}

fn ff4() {
    // macro m -> fn f -> let f;

    {
        let a0: FnF = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
    {
        let a1: FnF = m!();
    }
    fn f() -> FnF { 42 }
    {
        let a2: FnF = m!();
    }
    let f = || -> BindingF { 42 };
    {
        let a3: FnF = m!();
    }
}

fn ff5() {
    // macro m -> let f -> fn f;

    {
        let a0: FnF = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
    {
        let a1: FnF = m!();
    }
    let f = || -> BindingF { 42 };
    {
        let a2: FnF = m!();
    }
    fn f() -> FnF { 42 }
    {
        let a3: FnF = m!();
    }
}

fn ff6() {
    // macro m6 -> let f -> fn f;
    let a0: FnF = crate::m6!();
    {
        #[macro_export]
        macro_rules! m6 { () => { f() } }
        let f = || -> BindingF { 42 };
        let a1: FnF = crate::m6!();
    }
    let a2: FnF = crate::m6!();
    fn f() -> FnF { 42 }
    let a3: FnF = crate::m6!();
}

fn f_with_macro_export0() {
    {
        let a: BindingF = 42;                                   //~ NOTE `a` is defined here
        let c0: BindingF = crate::m1!();
        {
            {
                {
                    {
                        let d0: BindingF = m1!();
                        let d1: BindingF = crate::m1!();

                        #[macro_export]
                        macro_rules! m1 { () => { a } }         //~ ERROR cannot find value `a` in this scope
                                                                //~| NOTE not found in this scope
                        use m1;

                        let d2: BindingF = m1!();
                        let d3: BindingF = crate::m1!();
                    }
                }

                let e0: BindingF = m0!();
                #[macro_export]
                macro_rules! m0 { () => { a } }
                use m0;
                let e1: BindingF = m0!();
            }
        }
        let c1: BindingF = crate::m1!();
    }
    crate::m1!();                                               //~ NOTE in this expansion of crate::m1!
}

fn f_with_macro_export2() {
    fn a() {};
    {
        let a: BindingF = 42;                                   //~ NOTE `a` is defined here
        let c0: BindingF = crate::m2!();
        {
            let d0: BindingF = m2!();
            let d1: BindingF = crate::m2!();
            #[macro_export]
            macro_rules! m2 { () => { a } }                     //~ ERROR cannot find value `a` in this scope
                                                                //~| NOTE not found in this scope
            use m2;
            let d2: BindingF = m2!();
            let d3: BindingF = crate::m2!();
        }
        let c1: BindingF = crate::m2!();
    }
    crate::m2!();                                               //~ NOTE in this expansion of crate::m2!
}

fn f_with_macro_export3() {
    crate::m3!();                               //~ NOTE in this expansion of crate::m3!
    {
        let a: BindingF = 42;                   //~ NOTE `a` is defined here
                                                //~| NOTE `a` is defined here
        #[macro_export]
        macro_rules! m3 { () => { a } }         //~ ERROR cannot find value `a` in this scope
                                                //~| ERROR cannot find value `a` in this scope
                                                //~| NOTE not found in this scope
                                                //~| NOTE not found in this scope

    }
    crate::m3!();                               //~ NOTE in this expansion of crate::m3!
}

fn f_with_macro_export4() {
    crate::m4!();                               //~ NOTE in this expansion of crate::m4!
    {
        {
            let a: BindingF = 42;               //~ NOTE `a` is defined here
                                                //~| NOTE `a` is defined here
            {
                #[macro_export]
                macro_rules! m4 { () => { a } } //~ ERROR cannot find value `a` in this scope
                                                //~| ERROR cannot find value `a` in this scope
                                                //~| NOTE not found in this scope
                                                //~| NOTE not found in this scope
            }
        }
    }
    crate::m4!();                               //~ NOTE in this expansion of crate::m4!
}


fn f_with_macro_export5() {
    crate::m5!();                               //~ NOTE in this expansion of crate::m5!
    {
        let a: BindingF = 42;                   //~ NOTE `a` is defined here
                                                //~| NOTE `a` is defined here
        {
            #[macro_export]
            macro_rules! m5 { () => { a } }     //~ ERROR cannot find value `a` in this scope
                                                //~| ERROR cannot find value `a` in this scope
                                                //~| NOTE not found in this scope
                                                //~| NOTE not found in this scope
        }
    }
    fn a() {};
    crate::m5!();                               //~ NOTE in this expansion of crate::m5!
}

fn main () {}
