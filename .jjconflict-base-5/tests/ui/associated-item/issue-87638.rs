//@ run-rustfix

trait Trait {
    const FOO: usize;

    type Target;
}

struct S;

impl Trait for S {
    const FOO: usize = 0;
    type Target = ();
}

fn main() {
    let _: <S as Trait>::Output; //~ ERROR cannot find associated type `Output` in trait `Trait`
                                 //~^ HELP maybe you meant this associated type

    let _ = <S as Trait>::BAR; //~ ERROR cannot find method or associated constant `BAR` in trait `Trait`
                               //~^ HELP maybe you meant this associated constant
}
