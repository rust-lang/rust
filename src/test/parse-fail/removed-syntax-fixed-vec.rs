// compile-flags: -Z parse-only

type v = [isize * 3]; //~ ERROR expected one of `!`, `(`, `+`, `::`, `;`, `<`, or `]`, found `*`
