#[forbid(managed_heap_memory)];

type foo = { //~ ERROR type uses managed
    x: @int
};

fn main() {
    let _x : foo = {x : @10};
    //~^ ERROR type uses managed
    //~^^ ERROR type uses managed
}
