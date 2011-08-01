// xfail-stage0
// xfail-pretty
// error-pattern:meh
use std;
import std::str;

fn main() { let str_var: str = "meh"; fail #fmt("%s", str_var); }