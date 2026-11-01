//@ check-pass
//@ run-rustfix

fn main() {
    println!("Hello {:.1}!", f = 0.02f32);
    //~^ WARNING named argument `f` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity

    println!("Hello {:1.1}!", f = 0.02f32);
    //~^ WARNING named argument `f` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity

    println!("Hello {}!", f = 0.02f32);
    //~^ WARNING named argument `f` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity

    println!("Hello { }!", f = 0.02f32);
    //~^ WARNING named argument `f` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity

    println!("Hello {  }!", f = 0.02f32);
    //~^ WARNING named argument `f` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity
}
