//@edition: 2024
#![feature(gen_blocks)]

fn main() {
    basic();
    iterate();
    movable_gen();
}

fn basic() {
    gen fn foo() -> i32 {
        yield 42;
        for i in 5..10 {
            if i % 2 == 0 {
                continue;
            }
            yield i * 2;
        }
    }

    let v = foo().collect::<Vec<_>>();
    assert_eq!(v, &[42, 10, 14, 18]);
}

fn iterate() {
    fn foo() -> impl Iterator<Item = u32> {
        gen {
            yield 42;
            for x in 3..6 {
                yield x
            }
        }
    }

    fn moved() -> impl Iterator<Item = u32> {
        let mut x = "foo".to_string();
        gen move {
            yield 42;
            if x == "foo" {
                return;
            }
            x.clear();
            for x in 3..6 {
                yield x
            }
        }
    }

    let mut iter = foo();
    assert_eq!(iter.next(), Some(42));
    assert_eq!(iter.next(), Some(3));
    assert_eq!(iter.next(), Some(4));
    assert_eq!(iter.next(), Some(5));
    assert_eq!(iter.next(), None);
    // `gen` blocks are fused
    assert_eq!(iter.next(), None);

    let mut iter = moved();
    assert_eq!(iter.next(), Some(42));
    assert_eq!(iter.next(), None);
}

/// Ensure a generator can reborrow from a reference it captured.
/// Regression test for <https://github.com/rust-lang/rust/issues/159443>.
pub fn movable_gen() {
    fn make_gen(r: &mut u8) -> impl Iterator<Item = u8> {
        gen move {
            let a = r;
            *a = 1;
            yield 1;
            *a = 2;
        }
    }

    let mut a = 1;
    let mut i = make_gen(&mut a);
    assert_eq!(i.next(), Some(1));
    let mut j = i;
    assert_eq!(j.next(), None);
}
