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
// let _3: Foo;
// ...
// let mut _8: Foo;
// ...
// let mut _10: Bar;
// scope 1 {
//     debug a => _3;
//     let _4: Bar;
//     scope 2 {
//         debug b => _4;
//     }
// }
// bb0: {
//     StorageLive(_3);
//     _3 = Foo(const 5i32,);
//     StorageLive(_4);
//     _4 = Bar(const 6i32,);
//     ...
//     _5 = yield(move _6) -> [resume: bb1, drop: bb5];
// }
// bb1: {
//     ...
//     StorageLive(_8);
//     _8 = move _3;
//     _7 = const take::<Foo>(move _8) -> [return: bb2, unwind: bb9];
// }
// bb2: {
//     StorageDead(_8);
//     StorageDead(_7);
//     StorageLive(_9);
//     StorageLive(_10);
//     _10 = move _4;
//     _9 = const take::<Bar>(move _10) -> [return: bb3, unwind: bb8];
// }
// bb3: {
//     StorageDead(_10);
//     StorageDead(_9);
//     ...
//     StorageDead(_4);
//     StorageDead(_3);
//     drop(_1) -> [return: bb4, unwind: bb11];
// }
// bb4: {
//     return;
// }
// bb5: {
//     ...
//     StorageDead(_4);
//     drop(_3) -> [return: bb6, unwind: bb12];
// }
// bb6: {
//     StorageDead(_3);
//     drop(_1) -> [return: bb7, unwind: bb11];
// }
// bb7: {
//     generator_drop;
// }
// bb8 (cleanup): {
//     StorageDead(_10);
//     StorageDead(_9);
//     goto -> bb10;
// }
// bb9 (cleanup): {
//     StorageDead(_8);
//     StorageDead(_7);
//     goto -> bb10;
// }
// bb10 (cleanup): {
//     StorageDead(_4);
//     StorageDead(_3);
//     drop(_1) -> bb11;
// }
// bb11 (cleanup): {
//     resume;
// }
// bb12 (cleanup): {
//     StorageDead(_3);
//     drop(_1) -> bb11;
// }

// END rustc.main-{{closure}}.StateTransform.before.mir
