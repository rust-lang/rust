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
//      _2 = std::option::Option<i32>::Some(const 42i32,);
//      FakeRead(ForMatchedPlace, _2);
//      _3 = discriminant(_2);
//      switchInt(move _3) -> [0isize: bb4, 1isize: bb2, otherwise: bb7];
//  }
//  bb1: {
//      resume;
//  }
//  bb2: {
//      falseEdges -> [real: bb9, imaginary: bb3]; //pre_binding1
//  }
//  bb3: {
//      falseEdges -> [real: bb12, imaginary: bb4]; //pre_binding2
//  }
//  bb4: {
//      falseEdges -> [real: bb13, imaginary: bb5]; //pre_binding3
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
//  bb8: {
//      ...
//      return;
//  }
//  bb9: { // binding1 and guard
//      StorageLive(_8);
//      _8 = &(((promoted[2]: std::option::Option<i32>) as Some).0: i32);
//      _4 = &shallow (promoted[1]: std::option::Option<i32>);
//      _5 = &(((promoted[0]: std::option::Option<i32>) as Some).0: i32);
//      StorageLive(_9);
//      _9 = const guard() -> [return: bb10, unwind: bb1];
//  }
//  bb10: {
//      FakeRead(ForMatchGuard, _4);
//      FakeRead(ForMatchGuard, _5);
//      switchInt(move _9) -> [false: bb6, otherwise: bb11];
//  }
//  bb11: {
//      StorageLive(_6);
//      _6 = ((_2 as Some).0: i32);
//      StorageLive(_10);
//      _10 = _6;
//      _1 = (const 1i32, move _10);
//      StorageDead(_10);
//      goto -> bb8;
//  }
//  bb12: {
//      StorageLive(_11);
//      _11 = ((_2 as Some).0: i32);
//      StorageLive(_12);
//      _12 = _11;
//      _1 = (const 2i32, move _12);
//      StorageDead(_12);
//      goto -> bb8;
//  }
//  bb13: {
//      _1 = (const 3i32, const 3i32);
//      goto -> bb8;
//  }
// END rustc.full_tested_match.QualifyAndPromoteConstants.after.mir
//
// START rustc.full_tested_match2.QualifyAndPromoteConstants.before.mir
//  bb0: {
//      ...
//      _2 = std::option::Option<i32>::Some(const 42i32,);
//      FakeRead(ForMatchedPlace, _2);
//      _3 = discriminant(_2);
//      switchInt(move _3) -> [0isize: bb3, 1isize: bb2, otherwise: bb7];
//  }
//  bb1: {
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
//      unreachable;
//  }
//  bb6: { // to pre_binding3 (can skip 2 since this is `Some`)
//      falseEdges -> [real: bb4, imaginary: bb3];
//  }
//  bb7: {
//      unreachable;
//  }
//  bb8: {
//      ...
//      return;
//  }
//  bb9: { // binding1 and guard
//      StorageLive(_8);
//      _8 = &((_2 as Some).0: i32);
//      _4 = &shallow _2;
//      _5 = &((_2 as Some).0: i32);
//      StorageLive(_9);
//      _9 = const guard() -> [return: bb10, unwind: bb1];
//  }
//  bb10: { // end of guard
//      FakeRead(ForMatchGuard, _4);
//      FakeRead(ForMatchGuard, _5);
//      switchInt(move _9) -> [false: bb6, otherwise: bb11];
//  }
//  bb11: { // arm1
//      StorageLive(_6);
//      _6 = ((_2 as Some).0: i32);
//      StorageLive(_10);
//      _10 = _6;
//      _1 = (const 1i32, move _10);
//      StorageDead(_10);
//      goto -> bb8;
//  }
//  bb12: { // arm2
//      _1 = (const 3i32, const 3i32);
//      goto -> bb8;
//  }
//  bb13: { // binding3 and arm3
//      StorageLive(_11);
//      _11 = ((_2 as Some).0: i32);
//      StorageLive(_12);
//      _12 = _11;
//      _1 = (const 2i32, move _12);
//      StorageDead(_12);
//      goto -> bb8;
//  }
// END rustc.full_tested_match2.QualifyAndPromoteConstants.before.mir
//
// START rustc.main.QualifyAndPromoteConstants.before.mir
// bb0: {
//     ...
//      _2 = std::option::Option<i32>::Some(const 1i32,);
//      FakeRead(ForMatchedPlace, _2);
//      _3 = discriminant(_2);
//      switchInt(move _3) -> [1isize: bb2, otherwise: bb3];
//  }
//  bb1: {
//      resume;
//  }
//  bb2: {
//      falseEdges -> [real: bb10, imaginary: bb3]; //pre_binding1
//  }
//  bb3: {
//      falseEdges -> [real: bb13, imaginary: bb4]; //pre_binding2
//  }
//  bb4: {
//      falseEdges -> [real: bb14, imaginary: bb5]; //pre_binding3
//  }
//  bb5: {
//      falseEdges -> [real: bb17, imaginary: bb6]; //pre_binding4
//  }
//  bb6: {
//      unreachable;
//  }
//  bb7: { // to pre_binding2
//      falseEdges -> [real: bb3, imaginary: bb3];
//  }
//  bb8: { // to pre_binding4
//      falseEdges -> [real: bb5, imaginary: bb5];
//  }
//  bb9: {
//      ...
//      return;
//  }
//  bb10: { // binding1: Some(w) if guard()
//      StorageLive(_9);
//      _9 = &((_2 as Some).0: i32);
//      _5 = &shallow _2;
//      _6 = &((_2 as Some).0: i32);
//      StorageLive(_10);
//      _10 = const guard() -> [return: bb11, unwind: bb1];
//  }
//  bb11: { //end of guard
//      FakeRead(ForMatchGuard, _5);
//      FakeRead(ForMatchGuard, _6);
//      switchInt(move _10) -> [false: bb7, otherwise: bb12];
//  }
//  bb12: { // set up bindings for arm1
//      StorageLive(_7);
//      _7 = ((_2 as Some).0: i32);
//      _1 = const 1i32;
//      goto -> bb9;
//  }
//  bb13: { // binding2 & arm2
//      StorageLive(_11);
//      _11 = _2;
//      _1 = const 2i32;
//      goto -> bb9;
//  }
//  bb14: { // binding3: Some(y) if guard2(y)
//      StorageLive(_14);
//      _14 = &((_2 as Some).0: i32);
//      _5 = &shallow _2;
//      _6 = &((_2 as Some).0: i32);
//      StorageLive(_15);
//      StorageLive(_16);
//      _16 = (*_14);
//      _15 = const guard2(move _16) -> [return: bb15, unwind: bb1];
//  }
//  bb15: { // end of guard2
//      StorageDead(_16);
//      FakeRead(ForMatchGuard, _5);
//      FakeRead(ForMatchGuard, _6);
//      switchInt(move _15) -> [false: bb8, otherwise: bb16];
//  }
//  bb16: { // binding4 & arm4
//      StorageLive(_12);
//      _12 = ((_2 as Some).0: i32);
//      _1 = const 3i32;
//      goto -> bb9;
//  }
//  bb17: {
//      StorageLive(_17);
//      _17 = _2;
//      _1 = const 4i32;
//      goto -> bb9;
//  }
// END rustc.main.QualifyAndPromoteConstants.before.mir
