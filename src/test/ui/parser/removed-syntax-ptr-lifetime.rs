// compile-flags: -Z parse-only

type bptr = &lifetime/isize; //~ ERROR expected one of `!`, `(`, `::`, `;`, or `<`, found `/`
