// error-pattern: Wrong number of type arguments

enum quux<T> { bar }

fn foo(c: quux) { assert (false); }

fn main() { fail; }
