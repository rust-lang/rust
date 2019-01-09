// Beginners write `mod.item` when they should write `mod::item`.
// This tests that we suggest the latter when we encounter the former.

pub mod a {
    pub const I: i32 = 1;

    pub fn f() -> i32 { 2 }

    pub mod b {
        pub const J: i32 = 3;

        pub fn g() -> i32 { 4 }
    }
}

fn h1() -> i32 {
    a.I
    //~^ ERROR expected value, found module `a`
}

fn h2() -> i32 {
    a.g()
    //~^ ERROR expected value, found module `a`
}

fn h3() -> i32 {
    a.b.J
    //~^ ERROR expected value, found module `a`
}

fn h4() -> i32 {
    a::b.J
    //~^ ERROR expected value, found module `a::b`
}

fn h5() {
    a.b.f();
    //~^ ERROR expected value, found module `a`
    let v = Vec::new();
    v.push(a::b);
    //~^ ERROR expected value, found module `a::b`
}

fn h6() -> i32 {
    a::b.f()
    //~^ ERROR expected value, found module `a::b`
}

fn h7() {
    a::b
    //~^ ERROR expected value, found module `a::b`
}

fn h8() -> i32 {
    a::b()
    //~^ ERROR expected function, found module `a::b`
}

fn main() {}
