//@ compile-flags:-D unknown-lints -D bogus -D dead_cod

//@ error-pattern:requested on the command line with `-D bogus`
//@ error-pattern:requested on the command line with `-D dead_cod`
//@ error-pattern:requested on the command line with `-D unknown-lints`
//@ error-pattern:did you mean: `dead_code`

fn main() { }

//~? ERROR unknown lint: `bogus`
//~? ERROR unknown lint: `dead_cod`
//~? ERROR unknown lint: `bogus`
//~? ERROR unknown lint: `dead_cod`
//~? ERROR unknown lint: `bogus`
//~? ERROR unknown lint: `dead_cod`
