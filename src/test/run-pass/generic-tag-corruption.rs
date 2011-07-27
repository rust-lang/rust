


// This causes memory corruption in stage0.
tag thing[K] { some(K); }

fn main() { let x = some("hi"); }