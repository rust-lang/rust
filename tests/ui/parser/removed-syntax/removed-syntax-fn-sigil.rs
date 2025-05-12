fn main() {
    let x: fn~() = || (); //~ ERROR missing parameters for function definition
    //~| ERROR expected one of `->`, `;`, or `=`, found `~`
}
