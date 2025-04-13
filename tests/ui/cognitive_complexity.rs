#![warn(clippy::cognitive_complexity)]
#![allow(
    clippy::eq_op,
    clippy::needless_borrows_for_generic_args,
    clippy::needless_return,
    clippy::nonminimal_bool,
    clippy::uninlined_format_args
)]

#[rustfmt::skip]
fn main() {
//~^ cognitive_complexity

    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
    if true {
        println!("a");
    }
}

#[clippy::cognitive_complexity = "1"]
fn kaboom() {
    //~^ cognitive_complexity

    let n = 0;
    'a: for i in 0..20 {
        'b: for j in i..20 {
            for k in j..20 {
                if k == 5 {
                    break 'b;
                }
                if j == 3 && k == 6 {
                    continue 'a;
                }
                if k == j {
                    continue;
                }
                println!("bake");
            }
        }
        println!("cake");
    }
}

fn bloo() {
    match 42 {
        0 => println!("hi"),
        1 => println!("hai"),
        2 => println!("hey"),
        3 => println!("hallo"),
        4 => println!("hello"),
        5 => println!("salut"),
        6 => println!("good morning"),
        7 => println!("good evening"),
        8 => println!("good afternoon"),
        9 => println!("good night"),
        10 => println!("bonjour"),
        11 => println!("hej"),
        12 => println!("hej hej"),
        13 => println!("greetings earthling"),
        14 => println!("take us to you leader"),
        15 | 17 | 19 | 21 | 23 | 25 | 27 | 29 | 31 | 33 => println!("take us to you leader"),
        35 | 37 | 39 | 41 | 43 | 45 | 47 | 49 | 51 | 53 => println!("there is no undefined behavior"),
        55 | 57 | 59 | 61 | 63 | 65 | 67 | 69 | 71 | 73 => println!("I know borrow-fu"),
        _ => println!("bye"),
    }
}

// Short circuiting operations don't increase the complexity of a function.
// Note that the minimum complexity of a function is 1.
#[clippy::cognitive_complexity = "1"]
fn lots_of_short_circuits() -> bool {
    true && false && true && false && true && false && true
}

#[clippy::cognitive_complexity = "1"]
fn lots_of_short_circuits2() -> bool {
    true || false || true || false || true || false || true
}

#[clippy::cognitive_complexity = "1"]
fn baa() {
    //~^ cognitive_complexity

    let x = || match 99 {
        //~^ cognitive_complexity
        0 => 0,
        1 => 1,
        2 => 2,
        4 => 4,
        6 => 6,
        9 => 9,
        _ => 42,
    };
    if x() == 42 {
        println!("x");
    } else {
        println!("not x");
    }
}

#[clippy::cognitive_complexity = "1"]
fn bar() {
    //~^ cognitive_complexity

    match 99 {
        0 => println!("hi"),
        _ => println!("bye"),
    }
}

#[test]
#[clippy::cognitive_complexity = "1"]
/// Tests are usually complex but simple at the same time. `clippy::cognitive_complexity` used to
/// give lots of false-positives in tests.
fn dont_warn_on_tests() {
    //~^ cognitive_complexity

    match 99 {
        0 => println!("hi"),
        _ => println!("bye"),
    }
}

#[clippy::cognitive_complexity = "1"]
fn barr() {
    //~^ cognitive_complexity

    match 99 {
        0 => println!("hi"),
        1 => println!("bla"),
        2 | 3 => println!("blub"),
        _ => println!("bye"),
    }
}

#[clippy::cognitive_complexity = "1"]
fn barr2() {
    //~^ cognitive_complexity

    match 99 {
        0 => println!("hi"),
        1 => println!("bla"),
        2 | 3 => println!("blub"),
        _ => println!("bye"),
    }
    match 99 {
        0 => println!("hi"),
        1 => println!("bla"),
        2 | 3 => println!("blub"),
        _ => println!("bye"),
    }
}

#[clippy::cognitive_complexity = "1"]
fn barrr() {
    //~^ cognitive_complexity

    match 99 {
        0 => println!("hi"),
        1 => panic!("bla"),
        2 | 3 => println!("blub"),
        _ => println!("bye"),
    }
}

#[clippy::cognitive_complexity = "1"]
fn barrr2() {
    //~^ cognitive_complexity

    match 99 {
        0 => println!("hi"),
        1 => panic!("bla"),
        2 | 3 => println!("blub"),
        _ => println!("bye"),
    }
    match 99 {
        0 => println!("hi"),
        1 => panic!("bla"),
        2 | 3 => println!("blub"),
        _ => println!("bye"),
    }
}

#[clippy::cognitive_complexity = "1"]
fn barrrr() {
    //~^ cognitive_complexity

    match 99 {
        0 => println!("hi"),
        1 => println!("bla"),
        2 | 3 => panic!("blub"),
        _ => println!("bye"),
    }
}

#[clippy::cognitive_complexity = "1"]
fn barrrr2() {
    //~^ cognitive_complexity

    match 99 {
        0 => println!("hi"),
        1 => println!("bla"),
        2 | 3 => panic!("blub"),
        _ => println!("bye"),
    }
    match 99 {
        0 => println!("hi"),
        1 => println!("bla"),
        2 | 3 => panic!("blub"),
        _ => println!("bye"),
    }
}

#[clippy::cognitive_complexity = "1"]
fn cake() {
    //~^ cognitive_complexity

    if 4 == 5 {
        println!("yea");
    } else {
        panic!("meh");
    }
    println!("whee");
}

#[clippy::cognitive_complexity = "1"]
pub fn read_file(input_path: &str) -> String {
    //~^ cognitive_complexity

    use std::fs::File;
    use std::io::{Read, Write};
    use std::path::Path;
    let mut file = match File::open(&Path::new(input_path)) {
        Ok(f) => f,
        Err(err) => {
            panic!("Can't open {}: {}", input_path, err);
        },
    };

    let mut bytes = Vec::new();

    match file.read_to_end(&mut bytes) {
        Ok(..) => {},
        Err(_) => {
            panic!("Can't read {}", input_path);
        },
    };

    match String::from_utf8(bytes) {
        Ok(contents) => contents,
        Err(_) => {
            panic!("{} is not UTF-8 encoded", input_path);
        },
    }
}

enum Void {}

#[clippy::cognitive_complexity = "1"]
fn void(void: Void) {
    //~^ cognitive_complexity

    if true {
        match void {}
    }
}

#[clippy::cognitive_complexity = "1"]
fn mcarton_sees_all() {
    panic!("meh");
    panic!("möh");
}

#[clippy::cognitive_complexity = "1"]
fn try_() -> Result<i32, &'static str> {
    match 5 {
        5 => Ok(5),
        _ => return Err("bla"),
    }
}

#[clippy::cognitive_complexity = "1"]
fn try_again() -> Result<i32, &'static str> {
    let _ = Ok(42)?;
    let _ = Ok(43)?;
    let _ = Ok(44)?;
    let _ = Ok(45)?;
    let _ = Ok(46)?;
    let _ = Ok(47)?;
    let _ = Ok(48)?;
    let _ = Ok(49)?;
    match 5 {
        5 => Ok(5),
        _ => return Err("bla"),
    }
}

#[clippy::cognitive_complexity = "1"]
fn early() -> Result<i32, &'static str> {
    return Ok(5);
    return Ok(5);
    return Ok(5);
    return Ok(5);
    return Ok(5);
    return Ok(5);
    return Ok(5);
    return Ok(5);
    return Ok(5);
}

#[rustfmt::skip]
#[clippy::cognitive_complexity = "1"]
fn early_ret() -> i32 {
//~^ cognitive_complexity

    let a = if true { 42 } else { return 0; };
    let a = if a < 99 { 42 } else { return 0; };
    let a = if a < 99 { 42 } else { return 0; };
    let a = if a < 99 { 42 } else { return 0; };
    let a = if a < 99 { 42 } else { return 0; };
    let a = if a < 99 { 42 } else { return 0; };
    let a = if a < 99 { 42 } else { return 0; };
    let a = if a < 99 { 42 } else { return 0; };
    let a = if a < 99 { 42 } else { return 0; };
    let a = if a < 99 { 42 } else { return 0; };
    let a = if a < 99 { 42 } else { return 0; };
    let a = if a < 99 { 42 } else { return 0; };
    match 5 {
        5 => 5,
        _ => return 6,
    }
}

#[clippy::cognitive_complexity = "1"]
fn closures() {
    let x = |a: i32, b: i32| -> i32 {
        //~^ cognitive_complexity

        if true {
            println!("moo");
        }

        a + b
    };
}

struct Moo;

#[clippy::cognitive_complexity = "1"]
impl Moo {
    fn moo(&self) {
        //~^ cognitive_complexity

        if true {
            println!("moo");
        }
    }
}

#[clippy::cognitive_complexity = "1"]
mod issue9300 {
    async fn a() {
        //~^ cognitive_complexity

        let a = 0;
        if a == 0 {}
    }

    pub struct S;
    impl S {
        pub async fn async_method() {
            //~^ cognitive_complexity

            let a = 0;
            if a == 0 {}
        }
    }
}

#[clippy::cognitive_complexity = "1"]
mod issue14422 {
    fn foo() {
        //~^ cognitive_complexity
        for _ in 0..10 {
            println!("hello there");
        }
    }

    fn bar() {
        //~^ cognitive_complexity
        for _ in 0..10 {
            println!("hello there");
        }
        return;
        return;
    }
}
