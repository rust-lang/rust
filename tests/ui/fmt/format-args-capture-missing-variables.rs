fn main() {
    format!("{} {foo} {} {bar} {}", 1, 2, 3);
    //~^ ERROR: cannot find value `foo` in this scope
    //~^^ ERROR: cannot find value `bar` in this scope

    format!("{foo}"); //~ ERROR: cannot find value `foo` in this scope

    format!("{valuea} {valueb}", valuea=5, valuec=7);
    //~^ ERROR cannot find value `valueb` in this scope
    //~^^ ERROR named argument never used

    format!(r##"

        {foo}

    "##);
    //~^^^ ERROR: cannot find value `foo` in this scope

    panic!("{foo} {bar}", bar=1); //~ ERROR: cannot find value `foo` in this scope
}
