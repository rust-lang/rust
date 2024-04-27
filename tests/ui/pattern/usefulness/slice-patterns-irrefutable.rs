//@ check-pass

fn main() {
    let s: &[bool] = &[true; 0];
    let s0: &[bool; 0] = &[];
    let s1: &[bool; 1] = &[false; 1];
    let s2: &[bool; 2] = &[false; 2];

    let [] = s0;
    let [_] = s1;
    let [_, _] = s2;

    let [..] = s;
    let [..] = s0;
    let [..] = s1;
    let [..] = s2;

    let [_, ..] = s1;
    let [.., _] = s1;
    let [_, ..] = s2;
    let [.., _] = s2;

    let [_, _, ..] = s2;
    let [_, .., _] = s2;
    let [.., _, _] = s2;
}
