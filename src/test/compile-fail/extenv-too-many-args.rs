// error-pattern:malformed #env call

fn main() { #env[~"one", ~"two"]; }
