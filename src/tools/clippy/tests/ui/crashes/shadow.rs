//@ check-pass

fn main() {
    let x: [i32; {
        let u = 2;
        4
    }] = [2; { 4 }];
}
