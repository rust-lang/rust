//@ compile-flags:-D unknown-lints -D bogus -D dead_cod

//@ error-pattern:unknown lint: `bogus`
//@ error-pattern:requested on the command line with `-D bogus`
//@ error-pattern:requested on the command line with `-D dead_cod`
//@ error-pattern:requested on the command line with `-D unknown-lints`
//@ error-pattern:did you mean: `dead_code`

fn main() { }
