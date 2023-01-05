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

    // FIXME(compiler-errors): This one is still unsatisfying,
    // and probably a case I could see someone typing by accident..
    let _: Wow<[12]>;
    //~^ ERROR expected type, found
    //~| ERROR type provided when a constant was expected
}
