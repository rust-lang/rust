fn main() {
    let mut v = vec![1];
    let mut f = || v.push(2);
    let _w = v; //~ ERROR: cannot move out of `v`

    f();
}
