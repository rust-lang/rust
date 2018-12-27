// compile-flags: -Z parse-only

struct S;

impl S {
    fn foo(&self, &str bar) {}
    //~^ ERROR expected one of `:` or `@`
    //~| HELP declare the type after the parameter binding
    //~| SUGGESTION <identifier>: <type>
}

fn baz(S quux, xyzzy: i32) {}
//~^ ERROR expected one of `:` or `@`
//~| HELP declare the type after the parameter binding
//~| SUGGESTION <identifier>: <type>

fn one(i32 a b) {}
//~^ ERROR expected one of `:` or `@`

fn pattern((i32, i32) (a, b)) {}
//~^ ERROR expected `:`

fn fizz(i32) {}
//~^ ERROR expected one of `:` or `@`

fn missing_colon(quux S) {}
//~^ ERROR expected one of `:` or `@`
//~| HELP declare the type after the parameter binding
//~| SUGGESTION <identifier>: <type>

fn main() {}
