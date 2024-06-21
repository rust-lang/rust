fn main() {
    format!("{} {foo} {} {bar} {}", 1, 2, 3);
    //~^ ERROR: cannot find value `foo`
    //~^^ ERROR: cannot find value `bar`

    format!("{foo}"); //~ ERROR: cannot find value `foo`

    format!("{valuea} {valueb}", valuea=5, valuec=7);
    //~^ ERROR cannot find value `valueb`
    //~^^ ERROR named argument never used

    format!(r##"

        {foo}

    "##);
    //~^^^ ERROR: cannot find value `foo`

    panic!("{foo} {bar}", bar=1); //~ ERROR: cannot find value `foo`
}
