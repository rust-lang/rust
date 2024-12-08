//@ check-pass
// Make sure unused parens lint doesn't emit a false positive.
// See https://github.com/rust-lang/rust/issues/88519
#![deny(unused_parens)]
#![feature(type_ascription)]

// binary ops are tested in issue-71290-unused-paren-binop.rs

mod call {
    fn noop() -> u8 { 0 }
    fn outside() -> u8 {
        ({ noop })()
    }
    fn inside() -> u8 {
        ({ noop }())
    }
    fn outside_match() -> u8 {
        (match noop { x => x })()
    }
    fn inside_match() -> u8 {
        (match noop { x => x }())
    }
    fn outside_if() -> u8 {
        (if false { noop } else { noop })()
    }
    fn inside_if() -> u8 {
        (if false { noop } else { noop }())
    }
}

mod casts {
    fn outside() -> u8 {
        ({ 0 }) as u8
    }
    fn inside() -> u8 {
        ({ 0 } as u8)
    }
    fn outside_match() -> u8 {
        (match 0 { x => x }) as u8
    }
    fn inside_match() -> u8 {
        (match 0 { x => x } as u8)
    }
    fn outside_if() -> u8 {
        (if false { 0 } else { 0 }) as u8
    }
    fn inside_if() -> u8 {
        (if false { 0 } else { 0 } as u8)
    }
}

mod typeascription {
    fn outside() -> u8 {
        type_ascribe!(({ 0 }), u8)
    }
    fn outside_match() -> u8 {
        type_ascribe!((match 0 { x => x }), u8)
    }
    fn outside_if() -> u8 {
        type_ascribe!((if false { 0 } else { 0 }), u8)
    }
}

mod index {
    fn outside(x: &[u8]) -> u8 {
        ({ x })[0]
    }
    fn inside(x: &[u8]) -> u8 {
        ({ x }[0])
    }
    fn outside_match(x: &[u8]) -> u8 {
        (match x { x => x })[0]
    }
    fn inside_match(x: &[u8]) -> u8 {
        (match x { x => x }[0])
    }
    fn outside_if(x: &[u8]) -> u8 {
        (if false { x } else { x })[0]
    }
    fn inside_if(x: &[u8]) -> u8 {
        (if false { x } else { x }[0])
    }
}

fn main() {}
