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
    //~^ ERROR expected value, found module `a`
    //~| HELP use the path separator
}

fn h2() -> i32 {
    a.g()
    //~^ ERROR expected value, found module `a`
    //~| HELP use the path separator
}

fn h3() -> i32 {
    a.b.J
    //~^ ERROR expected value, found module `a`
    //~| HELP use the path separator
}

fn h4() -> i32 {
    a::b.J
    //~^ ERROR expected value, found module `a::b`
    //~| HELP a constant with a similar name exists
    //~| HELP use the path separator
}

fn h5() {
    a.b.f();
    //~^ ERROR expected value, found module `a`
    //~| HELP use the path separator
    let v = Vec::new();
    v.push(a::b);
    //~^ ERROR expected value, found module `a::b`
    //~| HELP a constant with a similar name exists
}

fn h6() -> i32 {
    a::b.f()
    //~^ ERROR expected value, found module `a::b`
    //~| HELP a constant with a similar name exists
    //~| HELP use the path separator
}

fn h7() {
    a::b
    //~^ ERROR expected value, found module `a::b`
    //~| HELP a constant with a similar name exists
}

fn h8() -> i32 {
    a::b()
    //~^ ERROR expected function, found module `a::b`
    //~| HELP a constant with a similar name exists
}

macro_rules! module {
    () => {
        a
        //~^ ERROR expected value, found module `a`
        //~| ERROR expected value, found module `a`
    };
}

macro_rules! create {
    (method) => {
        a.f()
        //~^ ERROR expected value, found module `a`
        //~| HELP use the path separator
    };
    (field) => {
        a.f
        //~^ ERROR expected value, found module `a`
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
