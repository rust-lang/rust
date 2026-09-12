// Beginners write `mod.item` when they should write `mod::item`.
// This tests that we suggest the latter when we encounter the former.

pub mod a {
    pub const I: i32 = 1;

    pub fn f() -> i32 { 2 }

    pub mod b {
        pub const J: i32 = 3;

        pub fn g() -> i32 { 4 }
    }
}

fn h1() -> i32 {
    a.I
    //~^ ERROR cannot find value `a` in this scope
    //~| HELP use the path separator
}

fn h2() -> i32 {
    a.g()
    //~^ ERROR cannot find value `a` in this scope
    //~| HELP use the path separator
}

fn h3() -> i32 {
    a.b.J
    //~^ ERROR cannot find value `a` in this scope
    //~| HELP use the path separator
}

fn h4() -> i32 {
    a::b.J
    //~^ ERROR cannot find value `b` in module `a`
    //~| HELP a constant with a similar name exists
    //~| HELP use the path separator
}

fn h5() {
    a.b.f();
    //~^ ERROR cannot find value `a` in this scope
    //~| HELP use the path separator
    let v = Vec::new();
    v.push(a::b);
    //~^ ERROR cannot find value `b` in module `a`
    //~| HELP a constant with a similar name exists
}

fn h6() -> i32 {
    a::b.f()
    //~^ ERROR cannot find value `b` in module `a`
    //~| HELP a constant with a similar name exists
    //~| HELP use the path separator
}

fn h7() {
    a::b
    //~^ ERROR cannot find value `b` in module `a`
    //~| HELP a constant with a similar name exists
}

fn h8() -> i32 {
    a::b()
    //~^ ERROR cannot find function `b` in module `a`
    //~| HELP a constant with a similar name exists
}

macro_rules! module {
    () => {
        a
        //~^ ERROR cannot find value `a` in this scope
        //~| ERROR cannot find value `a` in this scope
    };
}

macro_rules! create {
    (method) => {
        a.f()
        //~^ ERROR cannot find value `a` in this scope
        //~| HELP use the path separator
    };
    (field) => {
        a.f
        //~^ ERROR cannot find value `a` in this scope
        //~| HELP use the path separator
    };
}

fn h9() {
    //
    // Note that if the receiver is a macro call, we do not want to suggest to replace
    // `.` with `::` as that would be a syntax error.
    // Since the receiver is a module and not a type, we cannot suggest to surround
    // it with angle brackets.
    //

    module!().g::<()>(); // no `help` here!

    module!().g; // no `help` here!

    //
    // Ensure that the suggestion is shown for expressions inside of macro definitions.
    //

    let _ = create!(method);
    let _ = create!(field);
}

fn main() {}
