struct Wow<const N: usize>;

fn main() {
    let _: Wow<if true {}>;
    //~^ ERROR invalid const generic expression
    //~| HELP expressions must be enclosed in braces to be used as const generic arguments
    let _: Wow<|| ()>;
    //~^ ERROR invalid const generic expression
    //~| HELP expressions must be enclosed in braces to be used as const generic arguments
    let _: Wow<A.b>;
    //~^ ERROR expected one of
    //~| HELP expressions must be enclosed in braces to be used as const generic arguments
    let _: Wow<A.0>;
    //~^ ERROR expected one of
    //~| HELP expressions must be enclosed in braces to be used as const generic arguments
    let _: Wow<[]>;
    //~^ ERROR expected type
    //~| HELP expressions must be enclosed in braces to be used as const generic arguments
    let _: Wow<[12]>;
    //~^ ERROR expected type
    //~| ERROR invalid const generic expression
    //~| HELP expressions must be enclosed in braces to be used as const generic arguments
    let _: Wow<[0, 1, 3]>;
    //~^ ERROR expected type
    //~| HELP expressions must be enclosed in braces to be used as const generic arguments
    let _: Wow<[0xff; 8]>;
    //~^ ERROR expected type
    //~| ERROR invalid const generic expression
    //~| HELP expressions must be enclosed in braces to be used as const generic arguments
    let _: Wow<[1, 2]>; // Regression test for issue #81698.
    //~^ ERROR expected type
    //~| HELP expressions must be enclosed in braces to be used as const generic arguments
    let _: Wow<&0>;
    //~^ ERROR expected type
    //~| HELP expressions must be enclosed in braces to be used as const generic arguments
    let _: Wow<("", 0)>;
    //~^ ERROR expected type
    //~| HELP expressions must be enclosed in braces to be used as const generic arguments
    let _: Wow<(1 + 2) * 3>;
    //~^ ERROR expected type
    //~| HELP expressions must be enclosed in braces to be used as const generic arguments
    // FIXME(fmease): This one is pretty bad.
    let _: Wow<!0>;
    //~^ ERROR expected one of
    //~| HELP you might have meant to end the type parameters here
}
