fn main() {
    (a_function_that_takes_an_array[0; 10]);
    //~^ ERROR expected one of `.`, `?`, `]`, or an operator, found `;`
}

fn a_function_that_takes_an_array(arg: [u8; 10]) {
    let _ = arg;
}
