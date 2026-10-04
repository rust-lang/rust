// check that we can't do funny things with wildcards.

fn a() {
    match 1 {
        _ + 1 => () //~ error: expected one of `=>`, `if`, or `|`, found `+`
    }
}

fn b() {
    match 2 {
        (_ % 4) => () //~ error: expected one of `)`, `,`, `if`, or `|`, found `%`
    }
}

fn c() {
    match 3 {
        _.x() => () //~ error: expected one of `=>`, `if`, or `|`, found `.`
    }
}

fn d() {
    match 4 {
        _..=4 => () //~ error: expected one of `=>`, `if`, or `|`, found `..=`
    }
}

fn e() {
    match 5 {
        .._ => () //~ error: expected one of `=>`, `if`, or `|`, found reserved identifier `_`
    }
}

fn f() {
    match 6 {
        0..._ => ()
        //~^ error: inclusive range with no end
        //~| error: expected one of `=>`, `if`, or `|`, found reserved identifier `_`
    }
}

fn g() {
    match 7 {
        (_ * 0)..5 => () //~ error: expected one of `)`, `,`, `if`, or `|`, found `*`
    }
}

fn h() {
    match 8 {
        ..(_) => () //~ error: expected one of `=>`, `if`, or `|`, found `(`
    }
}

fn i() {
    match 9 {
        4..=(2 + _) => ()
        //~^ error: expected a pattern range bound, found an expression
        //~| error: range pattern bounds cannot have parentheses
    }
}

fn main() {}
