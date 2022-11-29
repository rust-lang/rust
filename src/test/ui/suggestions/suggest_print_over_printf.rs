// Suggest print macro when user erroneously uses printf

fn main() {
    let x = 4;
    printf("%d", x);
    //~^ ERROR cannot find function `printf` in this scope
    //~| HELP you may have meant to use the `print` macro
}
