use std::borrow::Cow;

fn main() {
    let x = "";
    std::thread_local!(static A: &str = x);
    //~^ ERROR attempt to use a non-constant value in a constant [E0435]
}

fn test<'a>(x: &'a str) {
    std::thread_local!(static A: Cow<'a, str> = Cow::Borrowed(""));
    //~^ ERROR can't use generic parameters from outer item [E0401]
    //~| ERROR can't use generic parameters from outer item [E0401]
    //~| ERROR can't use generic parameters from outer item [E0401]
    //~| ERROR can't use generic parameters from outer item [E0401]
    //~| ERROR can't use generic parameters from outer item [E0401]

    // wow, aren't macros great?
}
