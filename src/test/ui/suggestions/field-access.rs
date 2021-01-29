struct A {
    b: B,
}

enum B {
    Fst,
    Snd,
}

fn main() {
    let a = A { b: B::Fst };
    if let B::Fst = a {};
    //~^ ERROR mismatched types [E0308]
    // note: you might have meant to use field `b` of type `B`
    match a {
        B::Fst => (),
        B::Snd => (),
    }
    //~^^^ ERROR mismatched types [E0308]
    // note: you might have meant to use field `b` of type `B`
    //~^^^^ ERROR mismatched types [E0308]
    // note: you might have meant to use field `b` of type `B`
}
