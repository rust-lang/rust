//@ edition:2021
//@ check-pass
#![warn(unused)]

struct Point {
    x: u32,
    y: u32,
}

fn test1() {
    let t = (String::from("Hello"), String::from("World"));

    let c = ||  {
        let (t1, t2) = t;
        //~^ WARN unused variable: `t1`
        //~| WARN unused variable: `t2`
    };

    c();
}

fn test2() {
    let t = (String::from("Hello"), String::from("World"));

    let c = ||  {
        let (t1, _) = t;
        //~^ WARN unused variable: `t1`
    };

    c();
}

fn test3() {
    let t = (String::from("Hello"), String::from("World"));

    let c = ||  {
        let (_, t2) = t;
        //~^ WARN unused variable: `t2`
    };

    c();
}

fn test4() {
    let t = (String::from("Hello"), String::from("World"));

    let c = ||  {
        let (_, _) = t;
    };

    c();
}

fn test5() {
    let t = (String::new(), String::new());
    let _c = ||  {
        let _a = match t {
            (t1, _) => t1,
        };
    };
}

fn test6() {
    let t = (String::new(), String::new());
    let _c = ||  {
        let _a = match t {
            (_, t2) => t2,
        };
    };
}

fn test7() {
    let t = (String::new(), String::new());
    let _c = ||  {
        let _a = match t {
            (t1, t2) => (t1, t2),
        };
    };
}

fn test8() {
    let x = 0;
    let tup = (1, 2);
    let p = Point { x: 10, y: 20 };

    let c = || {
        let _ = x;
        let Point { x, y } = p;
        //~^ WARN unused variable: `x`
        println!("{}", y);
        let (_, _) = tup;
    };

    c();
}

fn test9() {
    let _z = 9;
    let t = (String::from("Hello"), String::from("World"));

    let c = ||  {
        let (_, t) = t;
        println!("{}", t);
    };

    c();
}

fn main() {
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();
    test7();
    test8();
    test9();
}
