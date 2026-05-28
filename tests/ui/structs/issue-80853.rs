struct S;

fn repro_ref(thing: S) {
    thing(); //~ ERROR expected function, found `S`
}

fn main() {}
