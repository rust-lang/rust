#![allow(dead_code)]

pub struct A<T> {
    pub value: T,
}

impl<T> A<T> {
    pub fn first(&self) -> u32 {
        static VALUE: u32 = 5;
        VALUE
    }

    pub fn second(&self) -> u32 {
        static VALUE: u32 = 6;
        VALUE
    }
}

pub static FIRST: u32 = 10;
pub static SECOND: u32 = 10;
pub static LINKS: [&u32; 3] = [&FIRST, &SECOND, &FIRST];
pub static TEXT: &str = "alloc123 alloc1 a2";
pub static DATA: [u32; 3] = [10, 20, 30];
pub static OFFSETS: [&u32; 2] = [&DATA[1], &DATA[2]];

pub struct Node {
    pub next: &'static Node,
    pub value: u8,
}

pub static LEFT: Node = Node { next: &RIGHT, value: 1 };
pub static RIGHT: Node = Node { next: &LEFT, value: 2 };

pub fn left() -> &'static Node {
    &LEFT
}

pub fn links() -> &'static [&'static u32; 3] {
    &LINKS
}

pub fn offsets() -> &'static [&'static u32; 2] {
    &OFFSETS
}

pub fn text() -> &'static str {
    TEXT
}

pub const PAIR: (&u32, &u32) = (&FIRST, &SECOND);
pub static PAIRS: [(&u32, &u32); 2] = [PAIR, PAIR];

pub fn pairs() -> &'static [(&'static u32, &'static u32); 2] {
    &PAIRS
}
