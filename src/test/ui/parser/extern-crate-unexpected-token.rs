// compile-flags: -Z parse-only

extern crte foo; //~ ERROR expected one of `crate`, `fn`, or `{`, found `crte`
