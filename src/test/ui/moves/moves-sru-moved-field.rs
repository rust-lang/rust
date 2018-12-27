#![feature(box_syntax)]

type Noncopyable = Box<isize>;

struct Foo {
    copied: isize,
    moved: Box<isize>,
    noncopyable: Noncopyable
}

fn test0(f: Foo, g: Noncopyable, h: Noncopyable) {
    // just copy implicitly copyable fields from `f`, no moves:
    let _b = Foo {moved: box 1, noncopyable: g, ..f};
    let _c = Foo {moved: box 2, noncopyable: h, ..f};
}

fn test1(f: Foo, g: Noncopyable, h: Noncopyable) {
    // copying move-by-default fields from `f`, so move:
    let _b = Foo {noncopyable: g, ..f};
    let _c = Foo {noncopyable: h, ..f}; //~ ERROR use of moved value: `f.moved`
}

fn main() {}
