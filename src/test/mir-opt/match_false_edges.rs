// compile-flags: -Z borrowck=mir

fn guard() -> bool {
    false
}

fn guard2(_: i32) -> bool {
    true
}

// no_mangle to make sure this gets instantiated even in an executable.
#[no_mangle]
pub fn full_tested_match() {
    let _ = match Some(42) {
        Some(x) if guard() => (1, x),
        Some(y) => (2, y),
        None => (3, 3),
    };
}

// no_mangle to make sure this gets instantiated even in an executable.
#[no_mangle]
pub fn full_tested_match2() {
    let _ = match Some(42) {
        Some(x) if guard() => (1, x),
        None => (3, 3),
        Some(y) => (2, y),
    };
}

fn main() {
    let _ = match Some(1) {
        Some(_w) if guard() => 1,
        _x => 2,
        Some(y) if guard2(y) => 3,
        _z => 4,
    };
}

// END RUST SOURCE
//
// START rustc.full_tested_match.PromoteTemps.after.mir
//  bb0: {
//      ...
//      _2 = std::option::Option::<i32>::Some(const 42i32,);
//      FakeRead(ForMatchedPlace, _2);
//      _3 = discriminant(_2);
//      switchInt(move _3) -> [0isize: bb1, 1isize: bb2, otherwise: bb4];
//  }
//  bb1: { // pre_binding3 and arm3
//      _1 = (const 3i32, const 3i32);
//      goto -> bb10;
//  }
//  bb2: {
//      falseEdges -> [real: bb5, imaginary: bb3]; //pre_binding1
//  }
//  bb3: {
//      falseEdges -> [real: bb9, imaginary: bb1]; //pre_binding2
//  }
//  bb4: {
//      unreachable;
//  }
//  bb5: { // binding1 and guard
//      StorageLive(_6);
//      _11 = const full_tested_match::promoted[0];
//      _6 = &(((*_11) as Some).0: i32);
//      _4 = &shallow _2;
//      StorageLive(_7);
//      _7 = const guard() -> [return: bb6, unwind: bb11];
//  }
//  bb6: { // end of guard
//      switchInt(move _7) -> [false: bb8, otherwise: bb7];
//  }
//  bb7: { // arm1
//      StorageDead(_7);
//      FakeRead(ForMatchGuard, _4);
//      FakeRead(ForGuardBinding, _6);
//      StorageLive(_5);
//      _5 = ((_2 as Some).0: i32);
//      StorageLive(_8);
//      _8 = _5;
//      _1 = (const 1i32, move _8);
//      StorageDead(_8);
//      StorageDead(_5);
//      StorageDead(_6);
//      goto -> bb10;
//  }
//  bb8: { // to pre_binding2
//      StorageDead(_7);
//      StorageDead(_6);
//      goto -> bb3;
//  }
//  bb9: { // arm2
//      StorageLive(_9);
//      _9 = ((_2 as Some).0: i32);
//      StorageLive(_10);
//      _10 = _9;
//      _1 = (const 2i32, move _10);
//      StorageDead(_10);
//      StorageDead(_9);
//      goto -> bb10;
//  }
//  bb10: { // match exit
//      StorageDead(_2);
//      StorageDead(_1);
//      _0 = ();
//      return;
//  }
//  bb11 (cleanup): {
//      resume;
//  }
// END rustc.full_tested_match.PromoteTemps.after.mir
//
// START rustc.full_tested_match2.PromoteTemps.before.mir
//  bb0: {
//      ...
//      _2 = std::option::Option::<i32>::Some(const 42i32,);
//      FakeRead(ForMatchedPlace, _2);
//      _3 = discriminant(_2);
//      switchInt(move _3) -> [0isize: bb1, 1isize: bb2, otherwise: bb4];
//  }
//  bb1: {
//      falseEdges -> [real: bb9, imaginary: bb3];
//  }
//  bb2: {
//      falseEdges -> [real: bb5, imaginary: bb1];
//  }
//  bb3: { // binding3 and arm3
//      StorageLive(_9);
//      _9 = ((_2 as Some).0: i32);
//      StorageLive(_10);
//      _10 = _9;
//      _1 = (const 2i32, move _10);
//      StorageDead(_10);
//      StorageDead(_9);
//      goto -> bb10;
//  }
//  bb4: {
//      unreachable;
//  }
//  bb5: { // binding1 and guard
//      StorageLive(_6);
//      _6 = &((_2 as Some).0: i32);
//      _4 = &shallow _2;
//      StorageLive(_7);
//      _7 = const guard() -> [return: bb6, unwind: bb11];
//  }
//  bb6: { // end of guard
//      switchInt(move _7) -> [false: bb8, otherwise: bb7];
//  }
//  bb7: {
//      StorageDead(_7);
//      FakeRead(ForMatchGuard, _4);
//      FakeRead(ForGuardBinding, _6);
//      StorageLive(_5);
//      _5 = ((_2 as Some).0: i32);
//      StorageLive(_8);
//      _8 = _5;
//      _1 = (const 1i32, move _8);
//      StorageDead(_8);
//      StorageDead(_5);
//      StorageDead(_6);
//      goto -> bb10;
//  }
//  bb8: { // to pre_binding3 (can skip 2 since this is `Some`)
//      StorageDead(_7);
//      StorageDead(_6);
//      falseEdges -> [real: bb3, imaginary: bb1];
//  }
//  bb9: { // arm2
//      _1 = (const 3i32, const 3i32);
//      goto -> bb10;
//  }
//  bb10: {
//      StorageDead(_2);
//      StorageDead(_1);
//      _0 = ();
//      return;
//  }
//  bb11 (cleanup): {
//      resume;
//  }
// END rustc.full_tested_match2.PromoteTemps.before.mir
//
// START rustc.main.PromoteTemps.before.mir
//  bb0: {
//     ...
//      _2 = std::option::Option::<i32>::Some(const 1i32,);
//      FakeRead(ForMatchedPlace, _2);
//      _4 = discriminant(_2);
//      switchInt(move _4) -> [1isize: bb2, otherwise: bb1];
//  }
//  bb1: {
//      falseEdges -> [real: bb9, imaginary: bb4];
//  }
//  bb2: {
//      falseEdges -> [real: bb5, imaginary: bb1];
//  }
//  bb3: {
//      StorageLive(_14);
//      _14 = _2;
//      _1 = const 4i32;
//      StorageDead(_14);
//      goto -> bb14;
//  }
//  bb4: {
//      falseEdges -> [real: bb10, imaginary: bb3];
//  }
//  bb5: {
//      StorageLive(_7);
//      _7 = &((_2 as Some).0: i32);
//      _5 = &shallow _2;
//      StorageLive(_8);
//      _8 = const guard() -> [return: bb6, unwind: bb15];
//  }
//  bb6: { //end of guard1
//      switchInt(move _8) -> [false: bb8, otherwise: bb7];
//  }
//  bb7: {
//      StorageDead(_8);
//      FakeRead(ForMatchGuard, _5);
//      FakeRead(ForGuardBinding, _7);
//      StorageLive(_6);
//      _6 = ((_2 as Some).0: i32);
//      _1 = const 1i32;
//      StorageDead(_6);
//      StorageDead(_7);
//      goto -> bb14;
//  }
//  bb8: {
//      StorageDead(_8);
//      StorageDead(_7);
//      falseEdges -> [real: bb1, imaginary: bb1];
//  }
//  bb9: { // binding2 & arm2
//      StorageLive(_9);
//      _9 = _2;
//      _1 = const 2i32;
//      StorageDead(_9);
//      goto -> bb14;
//  }
//  bb10: { // binding3: Some(y) if guard2(y)
//      StorageLive(_11);
//      _11 = &((_2 as Some).0: i32);
//      _5 = &shallow _2;
//      StorageLive(_12);
//      StorageLive(_13);
//      _13 = (*_11);
//      _12 = const guard2(move _13) -> [return: bb11, unwind: bb15];
//  }
//  bb11: { // end of guard2
//      StorageDead(_13);
//      switchInt(move _12) -> [false: bb13, otherwise: bb12];
//  }
//  bb12: { // binding4 & arm4
//      StorageDead(_12);
//      FakeRead(ForMatchGuard, _5);
//      FakeRead(ForGuardBinding, _11);
//      StorageLive(_10);
//      _10 = ((_2 as Some).0: i32);
//      _1 = const 3i32;
//      StorageDead(_10);
//      StorageDead(_11);
//      goto -> bb14;
//  }
//  bb13: {
//      StorageDead(_12);
//      StorageDead(_11);
//      falseEdges -> [real: bb3, imaginary: bb3];
//  }
//  bb14: {
//      StorageDead(_2);
//      StorageDead(_1);
//      _0 = ();
//      return;
//  }
//  bb15 (cleanup): {
//     resume;
//  }
// END rustc.main.PromoteTemps.before.mir
