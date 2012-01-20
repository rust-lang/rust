


// This causes memory corruption in stage0.
enum thing<K> { some(K), }

fn main() { let x = some("hi"); }
