// when the next token is not a semicolon,
// we should suggest to use semicolon if recovery is allowed
// See issue #143828

fn main() {
    let x = 5;
    let b: [i32, 5];
    //~^ ERROR expected one of `!`, `(`, `+`, `::`, `;`, `<`, or `]`, found `,`
    //~| ERROR expected value, found builtin type `i32` [E0423]
    let a: [i32, ];
    //~^ ERROR expected one of `!`, `(`, `+`, `::`, `;`, `<`, or `]`, found `,`
    //~| ERROR expected value, found builtin type `i32` [E0423]
    let c: [i32, x];
    //~^ ERROR expected one of `!`, `(`, `+`, `::`, `;`, `<`, or `]`, found `,`
    //~| ERROR expected value, found builtin type `i32` [E0423]
    let e: [i32 5];
    //~^ ERROR expected one of `!`, `(`, `+`, `::`, `;`, `<`, or `]`, found `5`
}
