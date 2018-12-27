#![allow(dead_code)]

struct S;

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
trait T { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
impl S { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
impl T for S { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
static s: usize = 0;

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
const c: usize = 0;

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
mod m { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
extern "C" { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
type A = usize;

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
fn main() { }
