// https://github.com/rust-lang/rust/issues/66706
fn a() {
    [0; [|_: _ &_| ()].len()]
    //~^ ERROR expected one of `,` or `|`, found `&`
    //~| ERROR type annotations needed
}

fn b() {
    [0; [|f @ &ref _| {} ; 0 ].len() ];
    //~^ ERROR expected identifier, found reserved identifier `_`
}

fn c() {
    [0; [|&_: _ &_| {}; 0 ].len()]
    //~^ ERROR expected one of `,` or `|`, found `&`
    //~| ERROR type annotations needed
}

fn d() {
    [0; match [|f @ &ref _| () ] {} ]
    //~^ ERROR expected identifier, found reserved identifier `_`
}

fn main() {}
