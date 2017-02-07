#![feature(plugin)]
#![plugin(clippy)]

#[deny(assign_ops)]
#[allow(unused_assignments)]
fn main() {
    let mut i = 1i32;
    i += 2; //~ ERROR assign operation detected
    //~^ HELP replace it with
    //~| SUGGESTION i = i + 2
    i += 2 + 17; //~ ERROR assign operation detected
    //~^ HELP replace it with
    //~| SUGGESTION i = i + 2 + 17
    i -= 6; //~ ERROR assign operation detected
    //~^ HELP replace it with
    //~| SUGGESTION i = i - 6
    i -= 2 - 1;
    //~^ ERROR assign operation detected
    //~| HELP replace it with
    //~| SUGGESTION i = i - (2 - 1)
    i *= 5; //~ ERROR assign operation detected
    //~^ HELP replace it with
    //~| SUGGESTION i = i * 5
    i *= 1+5; //~ ERROR assign operation detected
    //~^ HELP replace it with
    //~| SUGGESTION i = i * (1+5)
    i /= 32; //~ ERROR assign operation detected
    //~^ HELP replace it with
    //~| SUGGESTION i = i / 32
    i /= 32 | 5; //~ ERROR assign operation detected
    //~^ HELP replace it with
    //~| SUGGESTION i = i / (32 | 5)
    i /= 32 / 5; //~ ERROR assign operation detected
    //~^ HELP replace it with
    //~| SUGGESTION i = i / (32 / 5)
    i %= 42; //~ ERROR assign operation detected
    //~^ HELP replace it with
    //~| SUGGESTION i = i % 42
    i >>= i; //~ ERROR assign operation detected
    //~^ HELP replace it with
    //~| SUGGESTION i = i >> i
    i <<= 9 + 6 - 7; //~ ERROR assign operation detected
    //~^ HELP replace it with
    //~| SUGGESTION i = i << (9 + 6 - 7)
    i += 1 << 5;
    //~^ ERROR assign operation detected
    //~| HELP replace it with
    //~| SUGGESTION i = i + (1 << 5)
}

#[allow(dead_code, unused_assignments)]
#[deny(assign_op_pattern)]
fn bla() {
    let mut a = 5;
    a = a + 1; //~ ERROR manual implementation of an assign operation
    //~^ HELP replace it with
    //~| SUGGESTION a += 1
    a = 1 + a; //~ ERROR manual implementation of an assign operation
    //~^ HELP replace it with
    //~| SUGGESTION a += 1
    a = a - 1; //~ ERROR manual implementation of an assign operation
    //~^ HELP replace it with
    //~| SUGGESTION a -= 1
    a = a * 99; //~ ERROR manual implementation of an assign operation
    //~^ HELP replace it with
    //~| SUGGESTION a *= 99
    a = 42 * a; //~ ERROR manual implementation of an assign operation
    //~^ HELP replace it with
    //~| SUGGESTION a *= 42
    a = a / 2; //~ ERROR manual implementation of an assign operation
    //~^ HELP replace it with
    //~| SUGGESTION a /= 2
    a = a % 5; //~ ERROR manual implementation of an assign operation
    //~^ HELP replace it with
    //~| SUGGESTION a %= 5
    a = a & 1; //~ ERROR manual implementation of an assign operation
    //~^ HELP replace it with
    //~| SUGGESTION a &= 1
    a = 1 - a;
    a = 5 / a;
    a = 42 % a;
    a = 6 << a;
    let mut s = String::new();
    s = s + "bla";
}
