// This code caused a "no close delim when reparsing Expr" ICE in #139248.

thread_local! { static a : () = (if b) }
//~^ error: expected `{`, found `)`
//~| error: expected `{`, found `)`
//~| error: expected `{`, found `)`
//~| error: expected expression, found end of macro arguments

fn main() {}
