extern crate test;
use std::cell::Cell;

use test::Bencher;

use super::TypedArena;

#[allow(dead_code)]
#[derive(Debug, Eq, PartialEq)]
struct Point {
    x: i32,
    y: i32,
    z: i32,
}

impl<T> TypedArena<T> {
    /// Clears the arena. Deallocates all but the longest chunk which may be reused.
    fn clear(&mut self) {
        unsafe {
            // Clear the last chunk, which is partially filled.
            let mut chunks_borrow = self.chunks.borrow_mut();
            if let Some(mut last_chunk) = chunks_borrow.last_mut() {
                self.clear_last_chunk(&mut last_chunk);
                let len = chunks_borrow.len();
                // If `T` is ZST, code below has no effect.
                for mut chunk in chunks_borrow.drain(..len - 1) {
                    chunk.destroy(chunk.entries);
                }
            }
        }
    }
}

#[test]
fn test_unused() {
    let arena: TypedArena<Point> = TypedArena::default();
    assert!(arena.chunks.borrow().is_empty());
}

#[test]
fn test_arena_alloc_nested() {
    struct Inner {
        value: u8,
    }
    struct Outer<'a> {
        inner: &'a Inner,
    }
    enum EI<'e> {
        I(Inner),
        O(Outer<'e>),
    }

    struct Wrap<'a>(TypedArena<EI<'a>>);

    impl<'a> Wrap<'a> {
        fn alloc_inner<F: Fn() -> Inner>(&self, f: F) -> &Inner {
            match self.0.alloc(EI::I(f())) {
                EI::I(i) => i,
                _ => panic!("mismatch"),
            }
        }
        fn alloc_outer<F: Fn() -> Outer<'a>>(&self, f: F) -> &Outer<'_> {
            match self.0.alloc(EI::O(f())) {
                EI::O(o) => o,
                _ => panic!("mismatch"),
            }
        }
    }

    let arena = Wrap(TypedArena::default());

    let result = arena.alloc_outer(|| Outer { inner: arena.alloc_inner(|| Inner { value: 10 }) });

    assert_eq!(result.inner.value, 10);
}

#[test]
fn test_copy() {
    let arena = TypedArena::default();
    #[cfg(not(miri))]
    const N: usize = 100000;
    #[cfg(miri)]
    const N: usize = 1000;
    for _ in 0..N {
        arena.alloc(Point { x: 1, y: 2, z: 3 });
    }
}

#[bench]
fn bench_copy(b: &mut Bencher) {
    let arena = TypedArena::default();
    b.iter(|| arena.alloc(Point { x: 1, y: 2, z: 3 }))
}

#[bench]
fn bench_copy_nonarena(b: &mut Bencher) {
    b.iter(|| {
        let _: Box<_> = Box::new(Point { x: 1, y: 2, z: 3 });
    })
}

#[allow(dead_code)]
struct Noncopy {
    string: String,
    array: Vec<i32>,
}

#[test]
fn test_noncopy() {
    let arena = TypedArena::default();
    #[cfg(not(miri))]
    const N: usize = 100000;
    #[cfg(miri)]
    const N: usize = 1000;
    for _ in 0..N {
        arena.alloc(Noncopy { string: "hello world".to_string(), array: vec![1, 2, 3, 4, 5] });
    }
}

#[test]
fn test_typed_arena_zero_sized() {
    let arena = TypedArena::default();
    #[cfg(not(miri))]
    const N: usize = 100000;
    #[cfg(miri)]
    const N: usize = 1000;
    for _ in 0..N {
        arena.alloc(());
    }
}

#[test]
fn test_typed_arena_clear() {
    let mut arena = TypedArena::default();
    for _ in 0..10 {
        arena.clear();
        #[cfg(not(miri))]
        const N: usize = 10000;
        #[cfg(miri)]
        const N: usize = 100;
        for _ in 0..N {
            arena.alloc(Point { x: 1, y: 2, z: 3 });
        }
    }
}

#[bench]
fn bench_typed_arena_clear(b: &mut Bencher) {
    let mut arena = TypedArena::default();
    b.iter(|| {
        arena.alloc(Point { x: 1, y: 2, z: 3 });
        arena.clear();
    })
}

#[bench]
fn bench_typed_arena_clear_100(b: &mut Bencher) {
    let mut arena = TypedArena::default();
    b.iter(|| {
        for _ in 0..100 {
            arena.alloc(Point { x: 1, y: 2, z: 3 });
        }
        arena.clear();
    })
}

// Drop tests

struct DropCounter<'a> {
    count: &'a Cell<u32>,
}

impl Drop for DropCounter<'_> {
    fn drop(&mut self) {
        self.count.set(self.count.get() + 1);
    }
}

#[test]
fn test_typed_arena_drop_count() {
    let counter = Cell::new(0);
    {
        let arena: TypedArena<DropCounter<'_>> = TypedArena::default();
        for _ in 0..100 {
            // Allocate something with drop glue to make sure it doesn't leak.
            arena.alloc(DropCounter { count: &counter });
        }
    };
    assert_eq!(counter.get(), 100);
}

#[test]
fn test_typed_arena_drop_on_clear() {
    let counter = Cell::new(0);
    let mut arena: TypedArena<DropCounter<'_>> = TypedArena::default();
    for i in 0..10 {
        for _ in 0..100 {
            // Allocate something with drop glue to make sure it doesn't leak.
            arena.alloc(DropCounter { count: &counter });
        }
        arena.clear();
        assert_eq!(counter.get(), i * 100 + 100);
    }
}

thread_local! {
    static DROP_COUNTER: Cell<u32> = Cell::new(0)
}

struct SmallDroppable;

impl Drop for SmallDroppable {
    fn drop(&mut self) {
        DROP_COUNTER.with(|c| c.set(c.get() + 1));
    }
}

#[test]
fn test_typed_arena_drop_small_count() {
    DROP_COUNTER.with(|c| c.set(0));
    {
        let arena: TypedArena<SmallDroppable> = TypedArena::default();
        for _ in 0..100 {
            // Allocate something with drop glue to make sure it doesn't leak.
            arena.alloc(SmallDroppable);
        }
        // dropping
    };
    assert_eq!(DROP_COUNTER.with(|c| c.get()), 100);
}

#[bench]
fn bench_noncopy(b: &mut Bencher) {
    let arena = TypedArena::default();
    b.iter(|| {
        arena.alloc(Noncopy { string: "hello world".to_string(), array: vec![1, 2, 3, 4, 5] })
    })
}

#[bench]
fn bench_noncopy_nonarena(b: &mut Bencher) {
    b.iter(|| {
        let _: Box<_> =
            Box::new(Noncopy { string: "hello world".to_string(), array: vec![1, 2, 3, 4, 5] });
    })
}
