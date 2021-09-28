#![warn(clippy::many_single_char_names)]

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
            1 => println!(),
            e => panic!(),
        }
        match 5 {
            1 => println!(),
            _ => panic!(),
        }
    }
}

fn bindings(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32) {}

fn bindings2() {
    let (a, b, c, d, e, f, g, h): (bool, bool, bool, bool, bool, bool, bool, bool) = unimplemented!();
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

#[allow(clippy::many_single_char_names)]
fn issue_3198_allow_works() {
    let (a, b, c, d, e) = (0, 0, 0, 0, 0);
}

fn main() {}
