//@ check-pass
//@ run-rustfix

fn main() {
    println!("{b} {}", a=1, b=2);
    //~^ WARNING named argument `a` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity

    println!("{} {} {} {} {}", 0, a=1, b=2, c=3, d=4);
    //~^ WARNING named argument `a` is not used by name [named_arguments_used_positionally]
    //~| WARNING named argument `b` is not used by name [named_arguments_used_positionally]
    //~| WARNING named argument `c` is not used by name [named_arguments_used_positionally]
    //~| WARNING named argument `d` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity
    //~| HELP use the named argument by name to avoid ambiguity
    //~| HELP use the named argument by name to avoid ambiguity
    //~| HELP use the named argument by name to avoid ambiguity

    println!("Hello {:1$}!", "x", width = 5);
    //~^ WARNING named argument `width` is not used by name [named_arguments_used_positionally
    //~| HELP use the named argument by name to avoid ambiguity

    println!("Hello {:1$.2$}!", f = 0.02f32, width = 5, precision = 2);
    //~^ WARNING named argument `width` is not used by name [named_arguments_used_positionally
    //~| WARNING named argument `precision` is not used by name [named_arguments_used_positionally]
    //~| WARNING named argument `f` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity
    //~| HELP use the named argument by name to avoid ambiguity
    //~| HELP use the named argument by name to avoid ambiguity

    println!("Hello {0:1$.2$}!", f = 0.02f32, width = 5, precision = 2);
    //~^ WARNING named argument `width` is not used by name [named_arguments_used_positionally
    //~| WARNING named argument `precision` is not used by name [named_arguments_used_positionally]
    //~| WARNING named argument `f` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity
    //~| HELP use the named argument by name to avoid ambiguity
    //~| HELP use the named argument by name to avoid ambiguity

    println!(
        "{}, Hello {1:2$.3$} {4:5$.6$}! {1}",
        //~^ HELP use the named argument by name to avoid ambiguity
        //~| HELP use the named argument by name to avoid ambiguity
        //~| HELP use the named argument by name to avoid ambiguity
        //~| HELP use the named argument by name to avoid ambiguity
        //~| HELP use the named argument by name to avoid ambiguity
        //~| HELP use the named argument by name to avoid ambiguity
        //~| HELP use the named argument by name to avoid ambiguity
        1,
        f = 0.02f32,
        //~^ WARNING named argument `f` is not used by name [named_arguments_used_positionally]
        //~| WARNING named argument `f` is not used by name [named_arguments_used_positionally]
        width = 5,
        //~^ WARNING named argument `width` is not used by name [named_arguments_used_positionally
        precision = 2,
        //~^ WARNING named argument `precision` is not used by name [named_arguments_used_positionally]
        g = 0.02f32,
        //~^ WARNING named argument `g` is not used by name [named_arguments_used_positionally]
        width2 = 5,
        //~^ WARNING named argument `width2` is not used by name [named_arguments_used_positionally
        precision2 = 2
        //~^ WARNING named argument `precision2` is not used by name [named_arguments_used_positionally]
    );

    println!("Hello {:0.1}!", f = 0.02f32);
    //~^ WARNING named argument `f` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity

    println!("Hello {0:0.1}!", f = 0.02f32);
    //~^ WARNING named argument `f` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity

    println!("Hello {f:width$.precision$}!", f = 0.02f32, width = 5, precision = 2);

    let width = 5;
    let precision = 2;
    println!("Hello {f:width$.precision$}!", f = 0.02f32);

    let val = 5;
    println!("{:0$}", v = val);
    //~^ WARNING named argument `v` is not used by name [named_arguments_used_positionally]
    //~| WARNING named argument `v` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity
    //~| HELP use the named argument by name to avoid ambiguity
    println!("{0:0$}", v = val);
    //~^ WARNING named argument `v` is not used by name [named_arguments_used_positionally]
    //~| WARNING named argument `v` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity
    //~| HELP use the named argument by name to avoid ambiguity
    println!("{:0$.0$}", v = val);
    //~^ WARNING named argument `v` is not used by name [named_arguments_used_positionally]
    //~| WARNING named argument `v` is not used by name [named_arguments_used_positionally]
    //~| WARNING named argument `v` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity
    //~| HELP use the named argument by name to avoid ambiguity
    //~| HELP use the named argument by name to avoid ambiguity
    println!("{0:0$.0$}", v = val);
    //~^ WARNING named argument `v` is not used by name [named_arguments_used_positionally]
    //~| WARNING named argument `v` is not used by name [named_arguments_used_positionally]
    //~| WARNING named argument `v` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity
    //~| HELP use the named argument by name to avoid ambiguity
    //~| HELP use the named argument by name to avoid ambiguity

    println!("{} {a} {0}", a = 1);
    //~^ WARNING named argument `a` is not used by name [named_arguments_used_positionally]
    //~| WARNING named argument `a` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity
    //~| HELP use the named argument by name to avoid ambiguity

    println!("aaaaaaaaaaaaaaa\
                {:1$.2$}",
             //~^ HELP use the named argument by name to avoid ambiguity
             //~| HELP use the named argument by name to avoid ambiguity
             //~| HELP use the named argument by name to avoid ambiguity
             a = 1.0, b = 1, c = 2,
             //~^ WARNING named argument `a` is not used by name [named_arguments_used_positionally]
             //~| WARNING named argument `b` is not used by name [named_arguments_used_positionally]
             //~| WARNING named argument `c` is not used by name [named_arguments_used_positionally]
    );

    println!("aaaaaaaaaaaaaaa\
                {0:1$.2$}",
             //~^ HELP use the named argument by name to avoid ambiguity
             //~| HELP use the named argument by name to avoid ambiguity
             //~| HELP use the named argument by name to avoid ambiguity
             a = 1.0, b = 1, c = 2,
             //~^ WARNING named argument `a` is not used by name [named_arguments_used_positionally]
             //~| WARNING named argument `b` is not used by name [named_arguments_used_positionally]
             //~| WARNING named argument `c` is not used by name [named_arguments_used_positionally]
    );

    println!("{{{:1$.2$}}}", x = 1.0, width = 3, precision = 2);
    //~^ WARNING named argument `x` is not used by name [named_arguments_used_positionally]
    //~| WARNING named argument `width` is not used by name [named_arguments_used_positionally]
    //~| WARNING named argument `precision` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity
    //~| HELP use the named argument by name to avoid ambiguity
    //~| HELP use the named argument by name to avoid ambiguity
}
