//@ edition:2018
// issue#95237

#![feature(decl_macro)]

type FnF = i8;
type BindingF = i16;

fn f_without_definition_f() {
    let f = || -> BindingF { 42 };
    let a: BindingF = m!();
    macro m() { f() }
}

fn f_without_closure_f() {
    fn f() -> FnF { 42 }
    let a: FnF = m!();
    macro m() { f() }
}

fn f0() {
    // let f -> macro m -> fn f

    let a0: BindingF = m!();        //~ NOTE in this expansion of m!
    let f = || -> BindingF { 42 };  //~ NOTE `f` is defined here
    let a1: BindingF = m!();
    macro m() { f() }               //~ ERROR cannot find function `f` in this scope
                                    //~| NOTE not found in this scope
    let a2: BindingF = m!();
    fn f() -> FnF { 42 }            //~ NOTE you might have meant to refer to this function
    let a3: BindingF = m!();
}

fn f1() {
    // let f -> fn f -> macro m

    let a0: BindingF = m!();        //~ NOTE in this expansion of m!
    let f = || -> BindingF { 42 };  //~ NOTE `f` is defined here
    let a1: BindingF = m!();
    fn f() -> FnF { 42 }            //~ NOTE you might have meant to refer to this function
    let a2: BindingF = m!();
    macro m() { f() }               //~ ERROR cannot find function `f` in this scope
                                    //~| NOTE not found in this scope
    let a3: BindingF = m!();
}

fn f2() {
    // fn f -> let f -> macro m

    let a0: BindingF = m!();         //~ NOTE in this expansion of m!
    fn f() -> FnF { 42 }             //~ NOTE you might have meant to refer to this function
                                     //~| NOTE you might have meant to refer to this function
    let a1: BindingF = m!();         //~ NOTE in this expansion of m!
    let f = || -> BindingF { 42 };   //~ NOTE `f` is defined here
                                     //~| NOTE `f` is defined here
    let a2: BindingF = m!();
    macro m() { f() }               //~ ERROR cannot find function `f` in this scope
                                    //~| ERROR cannot find function `f` in this scope
                                    //~| NOTE not found in this scope
                                    //~| NOTE not found in this scope
    let a3: BindingF = m!();
}

fn f3() {
    // fn f -> macro m -> let f

    let a0: FnF = m!();
    fn f() -> FnF { 42 }
    let a1: FnF = m!();
    macro m() { f() }
    let a2: FnF = m!();
    let f = || -> BindingF { 42 };
    let a3: FnF = m!();
}

fn f4() {
    // macro m -> fn f -> let f;

    let a0: FnF = m!();
    macro m() { f() }
    let a1: FnF = m!();
    fn f() -> FnF { 42 }
    let a2: FnF = m!();
    let f = || -> BindingF { 42 };
    let a3: FnF = m!();
}

fn f5() {
    // macro m -> let f -> fn f;

    let a0: FnF = m!();
    macro m() { f() }
    let a1: FnF = m!();
    let f = || -> BindingF { 42 };
    let a2: FnF = m!();
    fn f() -> FnF { 42 }
    let a3: FnF = m!();
}

fn main () {}
