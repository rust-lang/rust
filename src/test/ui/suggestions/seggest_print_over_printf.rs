// Suggest to a user to use the print macros
// instead to use the printf.

fn main() {
    let x = 4;
    printf("%d", x);
    //~^ ERROR cannot find function `printf` in this scope
    //~| HELP you may have meant to use the `print` macro
}
