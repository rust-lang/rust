// compile-flags: -Z parse-only

fn f<T: ?>() {} //~ ERROR expected identifier, found `>`
