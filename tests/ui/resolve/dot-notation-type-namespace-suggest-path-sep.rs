// see also https://github.com/rust-lang/rust/issues/22692

type Alias = Vec<u32>;

mod foo {
    fn bar() {}
}

fn main() {
    let _ = String.new();
    //~^ ERROR expected value, found struct `String`
    //~| HELP use the path separator

    let _ = String.default;
    //~^ ERROR expected value, found struct `String`
    //~| HELP use the path separator

    let _ = Vec::<()>.with_capacity(1);
    //~^ ERROR expected value, found struct `Vec`
    //~| HELP use the path separator

    let _ = Alias.new();
    //~^ ERROR expected value, found type alias `Alias`
    //~| HELP use the path separator

    let _ = Alias.default;
    //~^ ERROR expected value, found type alias `Alias`
    //~| HELP use the path separator

    let _ = foo.bar;
    //~^ ERROR expected value, found module `foo`
    //~| HELP use the path separator
}

macro_rules! Type {
    () => {
        ::std::cell::Cell
        //~^ ERROR expected value, found struct `std::cell::Cell`
        //~| ERROR expected value, found struct `std::cell::Cell`
        //~| ERROR expected value, found struct `std::cell::Cell`
    };
    (alias) => {
        Alias
        //~^ ERROR expected value, found type alias `Alias`
        //~| ERROR expected value, found type alias `Alias`
        //~| ERROR expected value, found type alias `Alias`
    };
}

macro_rules! create {
    (type method) => {
        Vec.new()
        //~^ ERROR expected value, found struct `Vec`
        //~| HELP use the path separator
    };
    (type field) => {
        Vec.new
        //~^ ERROR expected value, found struct `Vec`
        //~| HELP use the path separator
    };
    (macro method) => {
        Type!().new(0)
        //~^ HELP use the path separator
    };
    (macro method alias) => {
        Type!(alias).new(0)
        //~^ HELP use the path separator
    };
}

macro_rules! check_ty {
    ($Ty:ident) => {
        $Ty.foo
        //~^ ERROR expected value, found type alias `Alias`
        //~| HELP use the path separator
    };
}
macro_rules! check_ident {
    ($Ident:ident) => {
        Alias.$Ident
        //~^ ERROR expected value, found type alias `Alias`
        //~| HELP use the path separator
    };
}
macro_rules! check_ty_ident {
    ($Ty:ident, $Ident:ident) => {
        $Ty.$Ident
        //~^ ERROR expected value, found type alias `Alias`
        //~| HELP use the path separator
    };
}

fn interaction_with_macros() {
    //
    // Verify that we do not only suggest to replace `.` with `::` if the receiver is a
    // macro call but that we also correctly suggest to surround it with angle brackets.
    //

    Type!().get();
    //~^ HELP use the path separator

    Type! {}.get;
    //~^ HELP use the path separator

    Type!(alias).get();
    //~^ HELP use the path separator

    Type! {alias}.get;
    //~^ HELP use the path separator

    //
    // Ensure that the suggestion is shown for expressions inside of macro definitions.
    //

    let _ = create!(type method);
    let _ = create!(type field);
    let _ = create!(macro method);
    let _ = create!(macro method alias);

    let _ = check_ty!(Alias);
    let _ = check_ident!(foo);
    let _ = check_ty_ident!(Alias, foo);
}
