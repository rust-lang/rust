macro_rules! falsy {
    () => { false };
}

fn foo() {
    if true {
    } else false {
    //~^ ERROR expected `{`, found keyword `false`
    }
}

fn foo2() {
    if true {
    } else falsy() {
    //~^ ERROR expected `{`, found `falsy`
    }
}

fn foo3() {
    if true {
    } else falsy();
    //~^ ERROR expected `{`, found `falsy`
}

fn foo4() {
    if true {
    } else loop{}
    //~^ ERROR expected `{`, found keyword `loop`
    {}
}

fn foo5() {
    if true {
    } else falsy!() {
    //~^ ERROR expected `{`, found `falsy`
    }
}

fn foo6() {
    if true {
    } else falsy!();
    //~^ ERROR expected `{`, found `falsy`
}

fn foo7() {
    if true {
    } else falsy! {} {
    //~^ ERROR expected `{`, found `falsy`
    }
}

fn foo8() {
    if true {
    } else falsy! {};
    //~^ ERROR expected `{`, found `falsy`
}

fn falsy() -> bool {
    false
}

fn main() {}
