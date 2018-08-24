// compile-flags: -Z parse-only

type v = [mut isize]; //~ ERROR expected type, found keyword `mut`
