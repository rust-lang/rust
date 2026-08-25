fn main() {
    // bad arguments to the format! call

    // bad number of arguments, see #44954 (originally #15780)

    format!("{}");
    //~^ ERROR: 1 positional argument in format string, but no arguments were given

    format!("{1}", 1);
    //~^ ERROR: invalid reference to positional argument 1 (there is 1 argument)
    //~^^ ERROR: argument never used

    format!("{} {}");
    //~^ ERROR: 2 positional arguments in format string, but no arguments were given

    format!("{0} {1}", 1);
    //~^ ERROR: invalid reference to positional argument 1 (there is 1 argument)

    format!("{0} {1} {2}", 1, 2);
    //~^ ERROR: invalid reference to positional argument 2 (there are 2 arguments)

    format!("{} {value} {} {}", 1, value=2);
    //~^ ERROR: 3 positional arguments in format string, but there are 2 arguments
    format!("{name} {value} {} {} {} {} {} {}", 0, name=1, value=2);
    //~^ ERROR: 6 positional arguments in format string, but there are 3 arguments

    format!("{} {foo} {} {bar} {}", 1, 2, 3);
    //~^ ERROR: cannot find value `foo` in this scope
    //~^^ ERROR: cannot find value `bar` in this scope

    format!("{foo}");                //~ ERROR: cannot find value `foo` in this scope
    format!("", 1, 2);               //~ ERROR: multiple unused formatting arguments
    format!("{}", 1, 2);             //~ ERROR: argument never used
    format!("{1}", 1, 2);            //~ ERROR: argument never used
    format!("{}", 1, foo=2);         //~ ERROR: named argument never used
    format!("{foo}", 1, foo=2);      //~ ERROR: argument never used
    format!("", foo=2);              //~ ERROR: named argument never used
    format!("{} {}", 1, 2, foo=1, bar=2);  //~ ERROR: multiple unused formatting arguments

    format!("{foo}", foo=1, foo=2);  //~ ERROR: duplicate argument
    format!("{foo} {} {}", foo=1, 2);   //~ ERROR: positional arguments cannot follow

    // bad named arguments, #35082

    format!("{valuea} {valueb}", valuea=5, valuec=7);
    //~^ ERROR cannot find value `valueb` in this scope
    //~^^ ERROR named argument never used

    // bad syntax of the format string

    format!("{"); //~ ERROR: expected `}` but string was terminated

    format!("foo } bar"); //~ ERROR: unmatched `}` found
    format!("foo }"); //~ ERROR: unmatched `}` found

    format!("foo %s baz", "bar"); //~ ERROR: argument never used

    format!(r##"

        {foo}

    "##);
    //~^^^ ERROR: cannot find value `foo` in this scope

    // bad syntax in format string with multiple newlines, #53836
    format!("first number: {}
second number: {}
third number: {}
fourth number: {}
fifth number: {}
sixth number: {}
seventh number: {}
eighth number: {}
ninth number: {
tenth number: {}",
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    //~^^ ERROR: invalid format string
    println!("{} {:.*} {}", 1, 3.2, 4);
    //~^ ERROR 4 positional arguments in format string, but there are 3 arguments
    //~| ERROR mismatched types
    println!("{} {:07$.*} {}", 1, 3.2, 4);
    //~^ ERROR invalid reference to positional arguments 3 and 7 (there are 3 arguments)
    //~| ERROR mismatched types
    println!("{} {:07$} {}", 1, 3.2, 4);
    //~^ ERROR invalid reference to positional argument 7 (there are 3 arguments)
    println!("{:foo}", 1); //~ ERROR unknown format trait `foo`
    println!("{5} {:4$} {6:7$}", 1);
    //~^ ERROR invalid reference to positional arguments 4, 5, 6 and 7 (there is 1 argument)
    let foo = 1;
    println!("{foo:0$}");
    //~^ ERROR invalid reference to positional argument 0 (no arguments were given)

    // We used to ICE here because we tried to unconditionally access the first argument, which
    // doesn't exist.
    println!("{:.*}");
    //~^ ERROR 2 positional arguments in format string, but no arguments were given
    println!("{:.0$}");
    //~^ ERROR invalid reference to positional argument 0 (no arguments were given)
}
