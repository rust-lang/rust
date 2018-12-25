// ignore-tidy-linelength

// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print simple
// gdbg-check:$1 = {x = 10, y = 20}
// gdbr-check:$1 = struct_with_destructor::WithDestructor {x: 10, y: 20}

// gdb-command:print noDestructor
// gdbg-check:$2 = {a = {x = 10, y = 20}, guard = -1}
// gdbr-check:$2 = struct_with_destructor::NoDestructorGuarded {a: struct_with_destructor::NoDestructor {x: 10, y: 20}, guard: -1}

// gdb-command:print withDestructor
// gdbg-check:$3 = {a = {x = 10, y = 20}, guard = -1}
// gdbr-check:$3 = struct_with_destructor::WithDestructorGuarded {a: struct_with_destructor::WithDestructor {x: 10, y: 20}, guard: -1}

// gdb-command:print nested
// gdbg-check:$4 = {a = {a = {x = 7890, y = 9870}}}
// gdbr-check:$4 = struct_with_destructor::NestedOuter {a: struct_with_destructor::NestedInner {a: struct_with_destructor::WithDestructor {x: 7890, y: 9870}}}


// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print simple
// lldbg-check:[...]$0 = WithDestructor { x: 10, y: 20 }
// lldbr-check:(struct_with_destructor::WithDestructor) simple = WithDestructor { x: 10, y: 20 }

// lldb-command:print noDestructor
// lldbg-check:[...]$1 = NoDestructorGuarded { a: NoDestructor { x: 10, y: 20 }, guard: -1 }
// lldbr-check:(struct_with_destructor::NoDestructorGuarded) noDestructor = NoDestructorGuarded { a: NoDestructor { x: 10, y: 20 }, guard: -1 }

// lldb-command:print withDestructor
// lldbg-check:[...]$2 = WithDestructorGuarded { a: WithDestructor { x: 10, y: 20 }, guard: -1 }
// lldbr-check:(struct_with_destructor::WithDestructorGuarded) withDestructor = WithDestructorGuarded { a: WithDestructor { x: 10, y: 20 }, guard: -1 }

// lldb-command:print nested
// lldbg-check:[...]$3 = NestedOuter { a: NestedInner { a: WithDestructor { x: 7890, y: 9870 } } }
// lldbr-check:(struct_with_destructor::NestedOuter) nested = NestedOuter { a: NestedInner { a: WithDestructor { x: 7890, y: 9870 } } }

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct NoDestructor {
    x: i32,
    y: i64
}

struct WithDestructor {
    x: i32,
    y: i64
}

impl Drop for WithDestructor {
    fn drop(&mut self) {}
}

struct NoDestructorGuarded {
    a: NoDestructor,
    guard: i64
}

struct WithDestructorGuarded {
    a: WithDestructor,
    guard: i64
}

struct NestedInner {
    a: WithDestructor
}

impl Drop for NestedInner {
    fn drop(&mut self) {}
}

struct NestedOuter {
    a: NestedInner
}


// The compiler adds a 'destructed' boolean field to structs implementing Drop. This field is used
// at runtime to prevent drop() to be executed more than once.
// This field must be incorporated by the debug info generation. Otherwise the debugger assumes a
// wrong size/layout for the struct.
fn main() {

    let simple = WithDestructor { x: 10, y: 20 };

    let noDestructor = NoDestructorGuarded {
        a: NoDestructor { x: 10, y: 20 },
        guard: -1
    };

    // If the destructor flag field is not incorporated into the debug info for 'WithDestructor'
    // then the debugger will have an invalid offset for the field 'guard' and thus should not be
    // able to read its value correctly (dots are padding bytes, D is the boolean destructor flag):
    //
    // 64 bit
    //
    // NoDestructorGuarded = 0000....00000000FFFFFFFF
    //                       <--------------><------>
    //                         NoDestructor   guard
    //
    //
    // withDestructorGuarded = 0000....00000000D.......FFFFFFFF
    //                         <--------------><------>          // How debug info says it is
    //                          WithDestructor  guard
    //
    //                         <----------------------><------>  // How it actually is
    //                              WithDestructor      guard
    //
    // 32 bit
    //
    // NoDestructorGuarded = 000000000000FFFFFFFF
    //                       <----------><------>
    //                       NoDestructor guard
    //
    //
    // withDestructorGuarded = 000000000000D...FFFFFFFF
    //                         <----------><------>      // How debug info says it is
    //                      WithDestructor  guard
    //
    //                         <--------------><------>  // How it actually is
    //                          WithDestructor  guard
    //
    let withDestructor = WithDestructorGuarded {
        a: WithDestructor { x: 10, y: 20 },
        guard: -1
    };

    // expected layout (64 bit) = xxxx....yyyyyyyyD.......D...
    //                            <--WithDestructor------>
    //                            <-------NestedInner-------->
    //                            <-------NestedOuter-------->
    let nested = NestedOuter { a: NestedInner { a: WithDestructor { x: 7890, y: 9870 } } };

    zzz(); // #break
}

fn zzz() {()}
