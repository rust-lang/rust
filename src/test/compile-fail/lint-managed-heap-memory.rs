#[forbid(managed_heap_memory)];

type Foo = { //~ ERROR type uses managed
    x: @int
};

fn main() {
    let _x : Foo = {x : @10};
    //~^ ERROR type uses managed
    //~^^ ERROR type uses managed
}
