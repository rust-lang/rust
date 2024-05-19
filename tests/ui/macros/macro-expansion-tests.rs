#![allow(unused_macros)]

mod macros_cant_escape_fns {
    fn f() {
        macro_rules! m { () => { 3 + 4 } }
    }
    fn g() -> i32 { m!() }
    //~^ ERROR cannot find macro
}

mod macros_cant_escape_mods {
    mod f {
        macro_rules! m { () => { 3 + 4 } }
    }
    fn g() -> i32 { m!() }
    //~^ ERROR cannot find macro
}

mod macros_can_escape_flattened_mods_test {
    #[macro_use]
    mod f {
        macro_rules! m { () => { 3 + 4 } }
    }
    fn g() -> i32 { m!() }
}

fn macro_tokens_should_match() {
    macro_rules! m { (a) => { 13 } }
    m!(a);
}

// should be able to use a bound identifier as a literal in a macro definition:
fn self_macro_parsing() {
    macro_rules! foo { (zz) => { 287; } }
    fn f(zz: i32) {
        foo!(zz);
    }
}

fn main() {}
