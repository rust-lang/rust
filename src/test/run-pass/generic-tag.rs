

tag option[T] { some(@T); none; }

fn main() { let option[int] a = some[int](@10); a = none[int]; }