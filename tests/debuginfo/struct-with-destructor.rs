//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print simple
// gdb-check:$1 = struct_with_destructor::WithDestructor {x: 10, y: 20}

// gdb-command:print noDestructor
// gdb-check:$2 = struct_with_destructor::NoDestructorGuarded {a: struct_with_destructor::NoDestructor {x: 10, y: 20}, guard: -1}

// gdb-command:print withDestructor
// gdb-check:$3 = struct_with_destructor::WithDestructorGuarded {a: struct_with_destructor::WithDestructor {x: 10, y: 20}, guard: -1}

// gdb-command:print nested
// gdb-check:$4 = struct_with_destructor::NestedOuter {a: struct_with_destructor::NestedInner {a: struct_with_destructor::WithDestructor {x: 7890, y: 9870}}}


// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:v simple
// lldb-check:[...] { x = 10 y = 20 }

// lldb-command:v noDestructor
// lldb-check:[...] { a = { x = 10 y = 20 } guard = -1 }

// lldb-command:v withDestructor
// lldb-check:[...] { a = { x = 10 y = 20 } guard = -1 }

// lldb-command:v nested
// lldb-check:[...] { a = { a = { x = 7890 y = 9870 } } }

#![allow(unused_variables)]

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
