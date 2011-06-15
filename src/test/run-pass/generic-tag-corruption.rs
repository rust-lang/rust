


// This causes memory corruption in stage0.
tag thing[K] { some(K); }

fn main() { auto x = some("hi"); }