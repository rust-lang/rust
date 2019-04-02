// compile-flags: -Z borrowck=mir

fn guard() -> bool {
    false
}

fn guard2(_:i32) -> bool {
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
// START rustc.full_tested_match.QualifyAndPromoteConstants.after.mir
//  bb0: {
//      ...
//      _2 = std::option::Option::<i32>::Some(const 42i32,);
//      FakeRead(ForMatchedPlace, _2);
//      _3 = discriminant(_2);
//      switchInt(move _3) -> [0isize: bb4, 1isize: bb2, otherwise: bb7];
//  }
//  bb1 (cleanup): {
//      resume;
//  }
//  bb2: {
//      falseEdges -> [real: bb8, imaginary: bb3]; //pre_binding1
//  }
//  bb3: {
//      falseEdges -> [real: bb11, imaginary: bb4]; //pre_binding2
//  }
//  bb4: {
//      falseEdges -> [real: bb12, imaginary: bb5]; //pre_binding3
//  }
//  bb5: {
//      unreachable;
//  }
//  bb6: { // to pre_binding2
//      falseEdges -> [real: bb3, imaginary: bb3];
//  }
//  bb7: {
//      unreachable;
//  }
//  bb8: { // binding1 and guard
//      StorageLive(_6);
//      _6 = &(((promoted[0]: std::option::Option<i32>) as Some).0: i32);
//      _4 = &shallow _2;
//      StorageLive(_7);
//      _7 = const guard() -> [return: bb9, unwind: bb1];
//  }
//  bb9: {
//      FakeRead(ForMatchGuard, _4);
//      FakeRead(ForGuardBinding, _6);
//      switchInt(move _7) -> [false: bb6, otherwise: bb10];
//  }
//  bb10: {
//      StorageLive(_5);
//      _5 = ((_2 as Some).0: i32);
//      StorageLive(_8);
//      _8 = _5;
//      _1 = (const 1i32, move _8);
//      StorageDead(_8);
//      goto -> bb13;
//  }
//  bb11: {
//      StorageLive(_9);
//      _9 = ((_2 as Some).0: i32);
//      StorageLive(_10);
//      _10 = _9;
//      _1 = (const 2i32, move _10);
//      StorageDead(_10);
//      goto -> bb13;
//  }
//  bb12: {
//      _1 = (const 3i32, const 3i32);
//      goto -> bb13;
//  }
//  bb13: {
//      ...
//      return;
//  }
// END rustc.full_tested_match.QualifyAndPromoteConstants.after.mir
//
// START rustc.full_tested_match2.QualifyAndPromoteConstants.before.mir
//  bb0: {
//      ...
//      _2 = std::option::Option::<i32>::Some(const 42i32,);
//      FakeRead(ForMatchedPlace, _2);
//      _3 = discriminant(_2);
//      switchInt(move _3) -> [0isize: bb3, 1isize: bb2, otherwise: bb7];
//  }
//  bb1 (cleanup): {
//      resume;
//  }
//  bb2: {
//      falseEdges -> [real: bb8, imaginary: bb3];
//  }
//  bb3: {
//      falseEdges -> [real: bb11, imaginary: bb4];
//  }
//  bb4: {
//      falseEdges -> [real: bb12, imaginary: bb5];
//  }
//  bb5: {
//      unreachable;
//  }
//  bb6: { // to pre_binding3 (can skip 2 since this is `Some`)
//      falseEdges -> [real: bb4, imaginary: bb3];
//  }
//  bb7: {
//      unreachable;
//  }
//  bb8: { // binding1 and guard
//      StorageLive(_6);
//      _6 = &((_2 as Some).0: i32);
//      _4 = &shallow _2;
//      StorageLive(_7);
//      _7 = const guard() -> [return: bb9, unwind: bb1];
//  }
//  bb9: { // end of guard
//      FakeRead(ForMatchGuard, _4);
//      FakeRead(ForGuardBinding, _6);
//      switchInt(move _7) -> [false: bb6, otherwise: bb10];
//  }
//  bb10: { // arm1
//      StorageLive(_5);
//      _5 = ((_2 as Some).0: i32);
//      StorageLive(_8);
//      _8 = _5;
//      _1 = (const 1i32, move _8);
//      StorageDead(_8);
//      goto -> bb13;
//  }
//  bb11: { // arm2
//      _1 = (const 3i32, const 3i32);
//      goto -> bb13;
//  }
//  bb12: { // binding3 and arm3
//      StorageLive(_9);
//      _9 = ((_2 as Some).0: i32);
//      StorageLive(_10);
//      _10 = _9;
//      _1 = (const 2i32, move _10);
//      StorageDead(_10);
//      goto -> bb13;
//  }
//  bb13: {
//      ...
//      return;
//  }
// END rustc.full_tested_match2.QualifyAndPromoteConstants.before.mir
//
// START rustc.main.QualifyAndPromoteConstants.before.mir
// bb0: {
//     ...
//      _2 = std::option::Option::<i32>::Some(const 1i32,);
//      FakeRead(ForMatchedPlace, _2);
//      _3 = discriminant(_2);
//      switchInt(move _3) -> [1isize: bb2, otherwise: bb3];
//  }
//  bb1 (cleanup): {
//      resume;
//  }
//  bb2: {
//      falseEdges -> [real: bb9, imaginary: bb3];
//  }
//  bb3: {
//      falseEdges -> [real: bb12, imaginary: bb4];
//  }
//  bb4: {
//      falseEdges -> [real: bb13, imaginary: bb5];
//  }
//  bb5: {
//      falseEdges -> [real: bb16, imaginary: bb6];
//  }
//  bb6: {
//      unreachable;
//  }
//  bb7: {
//      falseEdges -> [real: bb3, imaginary: bb3];
//  }
//  bb8: {
//      falseEdges -> [real: bb5, imaginary: bb5];
//  }
//  bb9: { // binding1: Some(w) if guard()
//      StorageLive(_7);
//      _7 = &((_2 as Some).0: i32);
//      _5 = &shallow _2;
//      StorageLive(_8);
//      _8 = const guard() -> [return: bb10, unwind: bb1];
//  }
//  bb10: { //end of guard
//      FakeRead(ForMatchGuard, _5);
//      FakeRead(ForGuardBinding, _7);
//      switchInt(move _8) -> [false: bb7, otherwise: bb11];
//  }
//  bb11: { // set up bindings for arm1
//      StorageLive(_6);
//      _6 = ((_2 as Some).0: i32);
//      _1 = const 1i32;
//      goto -> bb17;
//  }
//  bb12: { // binding2 & arm2
//      StorageLive(_9);
//      _9 = _2;
//      _1 = const 2i32;
//      goto -> bb17;
//  }
//  bb13: { // binding3: Some(y) if guard2(y)
//      StorageLive(_11);
//      _11 = &((_2 as Some).0: i32);
//      _5 = &shallow _2;
//      StorageLive(_12);
//      StorageLive(_13);
//      _13 = (*_11);
//      _12 = const guard2(move _13) -> [return: bb14, unwind: bb1];
//  }
//  bb14: { // end of guard2
//      StorageDead(_13);
//      FakeRead(ForMatchGuard, _5);
//      FakeRead(ForGuardBinding, _11);
//      switchInt(move _12) -> [false: bb8, otherwise: bb15];
//  }
//  bb15: { // binding4 & arm4
//      StorageLive(_10);
//      _10 = ((_2 as Some).0: i32);
//      _1 = const 3i32;
//      goto -> bb17;
//  }
//  bb16: {
//      StorageLive(_14);
//      _14 = _2;
//      _1 = const 4i32;
//      goto -> bb17;
//  }
//  bb17: {
//      ...
//      return;
//  }
// END rustc.main.QualifyAndPromoteConstants.before.mir
