//@ compile-flags:-D unknown-lints -D bogus -D dead_cod
//@ dont-require-annotations: HELP
//@ dont-require-annotations: NOTE

fn main() { }

//~? ERROR unknown lint: `bogus`
//~? ERROR unknown lint: `dead_cod`
//~? ERROR unknown lint: `bogus`
//~? ERROR unknown lint: `dead_cod`
//~? ERROR unknown lint: `bogus`
//~? ERROR unknown lint: `dead_cod`
//~? NOTE requested on the command line with `-D bogus`
//~? NOTE requested on the command line with `-D dead_cod`
//~? NOTE requested on the command line with `-D unknown-lints`
//~? HELP did you mean: `dead_code`
