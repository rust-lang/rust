//@compile-flags:-D bogus -D dead_cod

//@error-in-other-file:unknown lint: `bogus`
//@error-in-other-file:requested on the command line with `-D bogus`
//@error-in-other-file:unknown lint: `dead_cod`
//@error-in-other-file:requested on the command line with `-D dead_cod`
//@error-in-other-file:did you mean: `dead_code`

fn main() { }
