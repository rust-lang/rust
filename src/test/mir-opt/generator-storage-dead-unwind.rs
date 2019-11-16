// ignore-wasm32-bare compiled with panic=abort by default

// Test that we generate StorageDead on unwind paths for generators.
//
// Basic block and local names can safely change, but the StorageDead statements
// should not go away.

#![feature(generators, generator_trait)]

struct Foo(i32);

impl Drop for Foo {
    fn drop(&mut self) {}
}

struct Bar(i32);

fn take<T>(_x: T) {}

fn main() {
    let _gen = || {
        let a = Foo(5);
        let b = Bar(6);
        yield;
        take(a);
        take(b);
    };
}

// END RUST SOURCE

// START rustc.main-{{closure}}.StateTransform.before.mir
// ...
// let _2: Foo;
// ...
// let mut _7: Foo;
// ...
// let mut _9: Bar;
// scope 1 {
//     let _3: Bar;
//     scope 2 {
//     }
// }
// bb0: {
//     StorageLive(_2);
//     _2 = Foo(const 5i32,);
//     StorageLive(_3);
//     _3 = Bar(const 6i32,);
//     ...
//     _1 = suspend(move _5) -> [resume: bb1, drop: bb5];
// }
// bb1: {
//     ...
//     StorageLive(_6);
//     StorageLive(_7);
//     _7 = move _2;
//     _6 = const take::<Foo>(move _7) -> [return: bb2, unwind: bb11];
// }
// bb2: {
//     StorageDead(_7);
//     StorageDead(_6);
//     StorageLive(_8);
//     StorageLive(_9);
//     _9 = move _3;
//     _8 = const take::<Bar>(move _9) -> [return: bb3, unwind: bb10];
// }
// bb3: {
//     StorageDead(_9);
//     StorageDead(_8);
//     ...
//     StorageDead(_3);
//     StorageDead(_2);
//     drop(_1) -> [return: bb4, unwind: bb9];
// }
// bb4: {
//     return;
// }
// bb5: {
//     ...
//     StorageDead(_3);
//     drop(_2) -> [return: bb6, unwind: bb8];
// }
// bb6: {
//     StorageDead(_2);
//     drop(_1) -> [return: bb7, unwind: bb9];
// }
// bb7: {
//     generator_drop;
// }
// bb8 (cleanup): {
//     StorageDead(_2);
//     drop(_1) -> bb9;
// }
// bb9 (cleanup): {
//     resume;
// }
// bb10 (cleanup): {
//     StorageDead(_9);
//     StorageDead(_8);
//     goto -> bb12;
// }
// bb11 (cleanup): {
//     StorageDead(_7);
//     StorageDead(_6);
//     goto -> bb12;
// }
// bb12 (cleanup): {
//     StorageDead(_3);
//     StorageDead(_2);
//     drop(_1) -> bb9;
// }

// END rustc.main-{{closure}}.StateTransform.before.mir
