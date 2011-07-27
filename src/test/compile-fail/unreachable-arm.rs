// error-pattern:unreachable pattern

tag foo { a(@foo, int); b(uint); }

fn main() { alt b(1u) { b(_) | a(@_, 1) { } a(_, 1) { } } }