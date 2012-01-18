// error-pattern: Wrong number of type arguments

tag quux<T> { bar }

fn foo(c: quux) { assert (false); }

fn main() { fail; }
