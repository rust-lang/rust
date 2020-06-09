// run-rustfix

#![feature(or_patterns)]
#![feature(box_patterns)]
#![warn(clippy::unnested_or_patterns)]
#![allow(clippy::cognitive_complexity, clippy::match_ref_pats)]
#![allow(unreachable_patterns, irrefutable_let_patterns, unused_variables)]

fn main() {
    if let box 0 | box 2 = Box::new(0) {}
    if let box ((0 | 1)) | box (2 | 3) | box 4 = Box::new(0) {}
    const C0: &u8 = &1;
    if let &0 | C0 | &2 = &0 {}
    if let &mut 0 | &mut 2 = &mut 0 {}
    if let x @ 0 | x @ 2 = 0 {}
    if let (0, 1) | (0, 2) | (0, 3) = (0, 0) {}
    if let (1, 0) | (2, 0) | (3, 0) = (0, 0) {}
    if let (x, ..) | (x, 1) | (x, 2) = (0, 1) {}
    if let [0] | [1] = [0] {}
    if let [x, 0] | [x, 1] = [0, 1] {}
    if let [x, 0] | [x, 1] | [x, 2] = [0, 1] {}
    if let [x, ..] | [x, 1] | [x, 2] = [0, 1] {}
    struct TS(u8, u8);
    if let TS(0, x) | TS(1, x) = TS(0, 0) {}
    if let TS(1, 0) | TS(2, 0) | TS(3, 0) = TS(0, 0) {}
    if let TS(x, ..) | TS(x, 1) | TS(x, 2) = TS(0, 0) {}
    struct S {
        x: u8,
        y: u8,
    }
    if let S { x: 0, y } | S { y, x: 1 } = (S { x: 0, y: 1 }) {}
    if let S { x: 0, y, .. } | S { y, x: 1 } = (S { x: 0, y: 1 }) {}
}
