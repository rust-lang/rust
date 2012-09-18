#[forbid(owned_heap_memory)];

type Foo = { //~ ERROR type uses owned
    x: ~int
};

fn main() {
    let _x : Foo = {x : ~10};
    //~^ ERROR type uses owned
    //~^^ ERROR type uses owned
}
