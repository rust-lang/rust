// xfail-stage0
// error-pattern: Wrong number of type arguments

tag quux[T] { }

fn foo(c: quux) { assert (false); }

fn main() { fail; }