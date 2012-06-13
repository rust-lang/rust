// Issue #2482.

fn main() {
    let v1: [int] = [10, 20, 30,];
    let v2: [int] = [10, 20, 30];
    assert (v1[2] == v2[2]);
    let v3: [int] = [10,];
    let v4: [int] = [10];
    assert (v3[0] == v4[0]);
}
