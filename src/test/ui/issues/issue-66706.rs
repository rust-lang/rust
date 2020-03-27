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

fn main() {}
