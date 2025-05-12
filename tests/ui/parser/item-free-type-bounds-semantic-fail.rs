fn main() {}

fn semantics() {
    type A: Ord;
    //~^ ERROR bounds on `type`s in this context have no effect
    //~| ERROR free type alias without body
    type B: Ord = u8;
    //~^ ERROR bounds on `type`s in this context have no effect
    type C: Ord where 'static: 'static = u8;
    //~^ ERROR bounds on `type`s in this context have no effect
    type D<_T>: Ord;
    //~^ ERROR bounds on `type`s in this context have no effect
    //~| ERROR free type alias without body
    type E<_T>: Ord = u8;
    //~^ ERROR bounds on `type`s in this context have no effect
    //~| ERROR type parameter `_T` is never used
    type F<_T>: Ord where 'static: 'static = u8;
    //~^ ERROR bounds on `type`s in this context have no effect
    //~| ERROR type parameter `_T` is never used
}
