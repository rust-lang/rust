// when the next token is not a semicolon,
// we should suggest to use semicolon if recovery is allowed
// See issue #143828

fn main() {
    let x = 5;
    let b: [i32, 5];
    //~^ ERROR expected `;` or `]`, found `,`
    let a: [i32, ];
    //~^ ERROR expected `;` or `]`, found `,`
    //~| ERROR expected value, found builtin type `i32` [E0423]
    let c: [i32, x];
    //~^ ERROR expected `;` or `]`, found `,`
    //~| ERROR attempt to use a non-constant value in a constant [E0435]
    let e: [i32 5];
    //~^ ERROR expected `;` or `]`, found `5`
}
