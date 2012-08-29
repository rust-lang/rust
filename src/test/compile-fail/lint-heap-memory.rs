#[forbid(heap_memory)];

type foo = { //~ ERROR type uses managed
    x: @int
};

fn main() {
    let _x : { x : ~int } = {x : ~10};
    //~^ ERROR type uses owned
    //~^^ ERROR type uses owned
}
