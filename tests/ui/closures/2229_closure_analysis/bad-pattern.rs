// regression test for #108683
//@ edition:2021

enum Refutable {
    A,
    B,
}

fn example(v1: u32, v2: [u32; 4], v3: Refutable) {
    const PAT: u32 = 0;
    let v4 = &v2[..];
    || {
        let 0 = v1; //~ ERROR refutable pattern in local binding
        let (0 | 1) = v1; //~ ERROR refutable pattern in local binding
        let 1.. = v1; //~ ERROR refutable pattern in local binding
        let [0, 0, 0, 0] = v2; //~ ERROR refutable pattern in local binding
        let [0] = v4; //~ ERROR refutable pattern in local binding
        let Refutable::A = v3; //~ ERROR refutable pattern in local binding
        let PAT = v1; //~ ERROR refutable pattern in local binding
    };
}

fn main() {}
