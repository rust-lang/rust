fn a() {
    [0; [|_: _ &_| ()].len()]
    //~^ ERROR expected `,`, found `&`
    //~| ERROR type annotations needed
    //~| ERROR mismatched types
}

fn b() {
    [0; [|f @ &ref _| {} ; 0 ].len() ];
    //~^ ERROR expected identifier, found reserved identifier `_`
}

fn c() {
    [0; [|&_: _ &_| {}; 0 ].len()]
    //~^ ERROR expected `,`, found `&`
    //~| ERROR mismatched types
}

fn d() {
    [0; match [|f @ &ref _| () ] {} ]
    //~^ ERROR expected identifier, found reserved identifier `_`
    //~| ERROR `match` is not allowed in a `const`
    //~| ERROR mismatched types
}

fn main() {}
