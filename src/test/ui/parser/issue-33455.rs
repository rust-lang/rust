// compile-flags: -Z parse-only

use foo.bar; //~ ERROR expected one of `::`, `;`, or `as`, found `.`
