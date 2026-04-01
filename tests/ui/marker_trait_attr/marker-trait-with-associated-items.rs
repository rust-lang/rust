#![feature(marker_trait_attr)]
#![feature(associated_type_defaults)]

#[marker]
trait MarkerConst {
    const N: usize;
    //~^ ERROR marker traits cannot have associated items
}

#[marker]
trait MarkerType {
    type Output;
    //~^ ERROR marker traits cannot have associated items
}

#[marker]
trait MarkerFn {
    fn foo();
    //~^ ERROR marker traits cannot have associated items
}

#[marker]
trait MarkerConstWithDefault {
    const N: usize = 43;
    //~^ ERROR marker traits cannot have associated items
}

#[marker]
trait MarkerTypeWithDefault {
    type Output = ();
    //~^ ERROR marker traits cannot have associated items
}

#[marker]
trait MarkerFnWithDefault {
    fn foo() {}
    //~^ ERROR marker traits cannot have associated items
}

fn main() {}
