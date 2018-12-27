#![feature(optin_builtin_traits)]

// pp-exact

struct Test;

impl !Send for Test { }

pub fn main() { }
