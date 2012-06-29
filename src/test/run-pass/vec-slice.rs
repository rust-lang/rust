fn main() {
    let  v = ~[1,2,3,4,5];
    let v2 = vec::slice(v, 1, 3);
    assert (v2[0] == 2);
    assert (v2[1] == 3);
}
