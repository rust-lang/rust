// Fixed by #67160

trait Trait1 {
    type A;
}

trait Trait2 {
    type Type1<B>: Trait1<A=B>;
    //~^ ERROR: generic associated types are unstable
}

fn main() {}
