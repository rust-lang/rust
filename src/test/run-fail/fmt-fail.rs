// error-pattern:meh
// no-valgrind
use std;

fn main() { let str_var: istr = ~"meh"; fail #ifmt["%s", str_var]; }
