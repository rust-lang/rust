// see also https://github.com/rust-lang/rust/issues/22692

type Alias = Vec<u32>;

mod foo {
    fn bar() {}
}

fn main() {
    let _ = String.new();
    //~^ ERROR cannot find value `String` in this scope
    //~| HELP use the path separator

    let _ = String.default;
    //~^ ERROR cannot find value `String` in this scope
    //~| HELP use the path separator

    let _ = Vec::<()>.with_capacity(1);
    //~^ ERROR cannot find value `Vec` in this scope
    //~| HELP use the path separator

    let _ = Alias.new();
    //~^ ERROR cannot find value `Alias` in this scope
    //~| HELP use the path separator

    let _ = Alias.default;
    //~^ ERROR cannot find value `Alias` in this scope
    //~| HELP use the path separator

    let _ = foo.bar;
    //~^ ERROR cannot find value `foo` in this scope
    //~| HELP use the path separator
}

macro_rules! Type {
    () => {
        ::std::cell::Cell
        //~^ ERROR cannot find value `Cell` in module `::std::cell`
        //~| ERROR cannot find value `Cell` in module `::std::cell`
        //~| ERROR cannot find value `Cell` in module `::std::cell`
    };
    (alias) => {
        Alias
        //~^ ERROR cannot find value `Alias` in this scope
        //~| ERROR cannot find value `Alias` in this scope
        //~| ERROR cannot find value `Alias` in this scope
    };
}

macro_rules! create {
    (type method) => {
        Vec.new()
        //~^ ERROR cannot find value `Vec` in this scope
        //~| HELP use the path separator
    };
    (type field) => {
        Vec.new
        //~^ ERROR cannot find value `Vec` in this scope
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
        //~^ HELP use the path separator
    };
}
macro_rules! check_ident {
    ($Ident:ident) => {
        Alias.$Ident
        //~^ ERROR cannot find value `Alias` in this scope
        //~| HELP use the path separator
    };
}
macro_rules! check_ty_ident {
    ($Ty:ident, $Ident:ident) => {
        $Ty.$Ident
        //~^ HELP use the path separator
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
    //~^ ERROR cannot find value `Alias` in this scope
    let _ = check_ident!(foo);
    let _ = check_ty_ident!(Alias, foo);
    //~^ ERROR cannot find value `Alias` in this scope
}
