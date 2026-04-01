fn f() {
    S::<Item::<lol>::<nope>>;
}

fn g() {
    let _: Item::<lol>::<nope> = ();
}
