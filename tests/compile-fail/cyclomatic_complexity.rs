#![feature(plugin, custom_attribute)]
#![plugin(clippy)]
#![deny(clippy)]
#![deny(cyclomatic_complexity)]
#![allow(unused)]

fn main() { //~ ERROR: The function has a cyclomatic complexity of 28.
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

#[cyclomatic_complexity = "0"]
fn kaboom() {  //~ ERROR: The function has a cyclomatic complexity of 8
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

#[cyclomatic_complexity = "0"]
fn baa() { //~ ERROR: The function has a cyclomatic complexity of 2
    let x = || match 99 {
        0 => true,
        1 => false,
        2 => true,
        4 => true,
        6 => true,
        9 => true,
        _ => false,
    };
    if x() {
        println!("x");
    } else {
        println!("not x");
    }
}

#[cyclomatic_complexity = "0"]
fn bar() { //~ ERROR: The function has a cyclomatic complexity of 2
    match 99 {
        0 => println!("hi"),
        _ => println!("bye"),
    }
}

#[cyclomatic_complexity = "0"]
fn barr() { //~ ERROR: The function has a cyclomatic complexity of 2
    match 99 {
        0 => println!("hi"),
        1 => println!("bla"),
        2 | 3 => println!("blub"),
        _ => println!("bye"),
    }
}

#[cyclomatic_complexity = "0"]
fn barr2() { //~ ERROR: The function has a cyclomatic complexity of 3
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

#[cyclomatic_complexity = "0"]
fn barrr() { //~ ERROR: The function has a cyclomatic complexity of 2
    match 99 {
        0 => println!("hi"),
        1 => panic!("bla"),
        2 | 3 => println!("blub"),
        _ => println!("bye"),
    }
}

#[cyclomatic_complexity = "0"]
fn barrr2() { //~ ERROR: The function has a cyclomatic complexity of 3
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

#[cyclomatic_complexity = "0"]
fn barrrr() { //~ ERROR: The function has a cyclomatic complexity of 2
    match 99 {
        0 => println!("hi"),
        1 => println!("bla"),
        2 | 3 => panic!("blub"),
        _ => println!("bye"),
    }
}

#[cyclomatic_complexity = "0"]
fn barrrr2() { //~ ERROR: The function has a cyclomatic complexity of 3
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

#[cyclomatic_complexity = "0"]
fn cake() { //~ ERROR: The function has a cyclomatic complexity of 2
    if 4 == 5 {
        println!("yea");
    } else {
        panic!("meh");
    }
    println!("whee");
}


#[cyclomatic_complexity = "0"]
pub fn read_file(input_path: &str) -> String { //~ ERROR: The function has a cyclomatic complexity of 4
    use std::fs::File;
    use std::io::{Read, Write};
    use std::path::Path;
    let mut file = match File::open(&Path::new(input_path)) {
        Ok(f) => f,
        Err(err) => {
            panic!("Can't open {}: {}", input_path, err);
        }
    };

    let mut bytes = Vec::new();

    match file.read_to_end(&mut bytes) {
        Ok(..) => {},
        Err(_) => {
            panic!("Can't read {}", input_path);
        }
    };

    match String::from_utf8(bytes) {
        Ok(contents) => contents,
        Err(_) => {
            panic!("{} is not UTF-8 encoded", input_path);
        }
    }
}

enum Void {}

#[cyclomatic_complexity = "0"]
fn void(void: Void) { //~ ERROR: The function has a cyclomatic complexity of 1
    if true {
        match void {
        }
    }
}
