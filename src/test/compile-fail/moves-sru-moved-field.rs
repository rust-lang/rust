type Noncopyable = proc();

struct Foo {
    copied: int,
    moved: ~int,
    noncopyable: Noncopyable
}

fn test0(f: Foo, g: Noncopyable, h: Noncopyable) {
    // just copy implicitly copyable fields from `f`, no moves:
    let _b = Foo {moved: ~1, noncopyable: g, ..f};
    let _c = Foo {moved: ~2, noncopyable: h, ..f};
}

fn test1(f: Foo, g: Noncopyable, h: Noncopyable) {
    // copying move-by-default fields from `f`, so move:
    let _b = Foo {noncopyable: g, ..f};
    let _c = Foo {noncopyable: h, ..f}; //~ ERROR use of moved value: `f`
}

fn test2(f: Foo, g: Noncopyable) {
    // move non-copyable field
    let _b = Foo {copied: 22, moved: ~23, ..f};
    let _c = Foo {noncopyable: g, ..f}; //~ ERROR use of moved value: `f`
}

fn main() {}
