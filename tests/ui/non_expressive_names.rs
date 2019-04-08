#![warn(clippy::all)]
#![allow(unused, clippy::println_empty_string)]

#[derive(Clone, Debug)]
enum MaybeInst {
    Split,
    Split1(usize),
    Split2(usize),
}

struct InstSplit {
    uiae: usize,
}

impl MaybeInst {
    fn fill(&mut self) {
        let filled = match *self {
            MaybeInst::Split1(goto1) => panic!(1),
            MaybeInst::Split2(goto2) => panic!(2),
            _ => unimplemented!(),
        };
        unimplemented!()
    }
}

fn bla() {
    let a: i32;
    let (b, c, d): (i32, i64, i16);
    {
        {
            let cdefg: i32;
            let blar: i32;
        }
        {
            let e: i32;
        }
        {
            let e: i32;
            let f: i32;
        }
        match 5 {
            1 => println!(""),
            e => panic!(),
        }
        match 5 {
            1 => println!(""),
            _ => panic!(),
        }
    }
}

fn bindings(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32) {}

fn bindings2() {
    let (a, b, c, d, e, f, g, h) = unimplemented!();
}

fn shadowing() {
    let a = 0i32;
    let a = 0i32;
    let a = 0i32;
    let a = 0i32;
    let a = 0i32;
    let a = 0i32;
    {
        let a = 0i32;
    }
}

fn patterns() {
    enum Z {
        A(i32),
        B(i32),
        C(i32),
        D(i32),
        E(i32),
        F(i32),
    }

    // These should not trigger a warning, since the pattern bindings are a new scope.
    match Z::A(0) {
        Z::A(a) => {},
        Z::B(b) => {},
        Z::C(c) => {},
        Z::D(d) => {},
        Z::E(e) => {},
        Z::F(f) => {},
    }
}

fn underscores_and_numbers() {
    let _1 = 1; //~ERROR Consider a more descriptive name
    let ____1 = 1; //~ERROR Consider a more descriptive name
    let __1___2 = 12; //~ERROR Consider a more descriptive name
    let _1_ok = 1;
}

fn issue2927() {
    let args = 1;
    format!("{:?}", 2);
}

fn issue3078() {
    match "a" {
        stringify!(a) => {},
        _ => {},
    }
}

struct Bar;

impl Bar {
    fn bar() {
        let _1 = 1;
        let ____1 = 1;
        let __1___2 = 12;
        let _1_ok = 1;
    }
}

fn main() {}
