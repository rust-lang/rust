//@ dont-require-annotations: SUGGESTION

struct S;

impl S {
    fn foo(&self, &str bar) {}
    //~^ ERROR expected one of `:`, `@`
    //~| HELP declare the type after the parameter binding
    //~| SUGGESTION <identifier>: <type>
}

fn baz(S quux, xyzzy: i32) {}
//~^ ERROR expected one of `:`, `@`
//~| HELP declare the type after the parameter binding
//~| SUGGESTION <identifier>: <type>

fn one(i32 a b) {}
//~^ ERROR expected one of `:`, `@`

fn pattern((i32, i32) (a, b)) {}
//~^ ERROR expected one of `:`

fn fizz(i32) {}
//~^ ERROR expected one of `:`, `@`
//~| HELP if this is a parameter name, give it a type
//~| HELP if this is a type, explicitly ignore the parameter name

fn missing_colon(quux S) {}
//~^ ERROR expected one of `:`, `@`
//~| HELP declare the type after the parameter binding
//~| SUGGESTION <identifier>: <type>

fn main() {}
