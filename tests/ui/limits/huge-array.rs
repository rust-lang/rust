// build-fail

fn generic<T: Copy>(t: T) {
    let s: [T; 1518600000] = [t; 1518600000];
    //~^ ERROR values of the type `[[u8; 1518599999]; 1518600000]` are too big
}

fn main() {
    let x: [u8; 1518599999] = [0; 1518599999];
    generic::<[u8; 1518599999]>(x);
}
