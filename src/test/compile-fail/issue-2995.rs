fn bad (p: *int) {
    let _q: &int = p as &int; //~ ERROR non-scalar cast
}