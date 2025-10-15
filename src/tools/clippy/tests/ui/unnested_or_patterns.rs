#![feature(box_patterns)]
#![warn(clippy::unnested_or_patterns)]
#![allow(
    clippy::cognitive_complexity,
    clippy::match_ref_pats,
    clippy::upper_case_acronyms,
    clippy::needless_if,
    clippy::manual_range_patterns
)]
#![allow(unreachable_patterns, irrefutable_let_patterns, unused)]

struct S {
    x: u8,
    y: u8,
}

fn main() {
    // Should be ignored by this lint, as nesting requires more characters.
    if let &0 | &2 = &0 {}

    if let box 0 | box 2 = Box::new(0) {}
    //~^ unnested_or_patterns
    if let box ((0 | 1)) | box (2 | 3) | box 4 = Box::new(0) {}
    //~^ unnested_or_patterns
    const C0: Option<u8> = Some(1);
    if let Some(1) | C0 | Some(2) = None {}
    //~^ unnested_or_patterns
    if let &mut 0 | &mut 2 = &mut 0 {}
    //~^ unnested_or_patterns
    if let x @ 0 | x @ 2 = 0 {}
    //~^ unnested_or_patterns
    if let (0, 1) | (0, 2) | (0, 3) = (0, 0) {}
    //~^ unnested_or_patterns
    if let (1, 0) | (2, 0) | (3, 0) = (0, 0) {}
    //~^ unnested_or_patterns
    if let (x, ..) | (x, 1) | (x, 2) = (0, 1) {}
    //~^ unnested_or_patterns
    if let [0] | [1] = [0] {}
    //~^ unnested_or_patterns
    if let [x, 0] | [x, 1] = [0, 1] {}
    //~^ unnested_or_patterns
    if let [x, 0] | [x, 1] | [x, 2] = [0, 1] {}
    //~^ unnested_or_patterns
    if let [x, ..] | [x, 1] | [x, 2] = [0, 1] {}
    //~^ unnested_or_patterns
    struct TS(u8, u8);
    if let TS(0, x) | TS(1, x) = TS(0, 0) {}
    //~^ unnested_or_patterns
    if let TS(1, 0) | TS(2, 0) | TS(3, 0) = TS(0, 0) {}
    //~^ unnested_or_patterns
    if let TS(x, ..) | TS(x, 1) | TS(x, 2) = TS(0, 0) {}
    //~^ unnested_or_patterns
    if let S { x: 0, y } | S { y, x: 1 } = (S { x: 0, y: 1 }) {}
    //~^ unnested_or_patterns
    if let S { x: 0, y, .. } | S { y, x: 1 } = (S { x: 0, y: 1 }) {}
}

#[clippy::msrv = "1.52"]
fn msrv_1_52() {
    if let [1] | [52] = [0] {}
}

#[clippy::msrv = "1.53"]
fn msrv_1_53() {
    if let [1] | [53] = [0] {}
    //~^ unnested_or_patterns
}

mod issue9952 {
    fn or_in_local() {
        let (0 | (1 | _)) = 0;
        //~^ unnested_or_patterns

        if let (0 | (1 | _)) = 0 {}
        //~^ unnested_or_patterns
    }

    fn or_in_param((x | (x | x)): i32) {}
    //~^ unnested_or_patterns
}

fn issue15219() {
    struct Foo {
        x: u8,
    }

    // the original repro
    if let Foo { x } | Foo { x } = (Foo { x: 0 }) {}

    // also works with more fields
    if let S { x, y } | S { x, y } = (S { x: 0, y: 0 }) {}

    // `y` not triggering the lint doesn't stop the `x` from getting flagged
    if let S { y, x: 0 } | S { y, x: 1 } = (S { x: 0, y: 1 }) {}
    //~^ unnested_or_patterns
}
