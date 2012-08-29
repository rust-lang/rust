#[forbid(owned_heap_memory)];

type foo = { //~ ERROR type uses owned
    x: ~int
};

fn main() {
    let _x : foo = {x : ~10};
    //~^ ERROR type uses owned
    //~^^ ERROR type uses owned
}
