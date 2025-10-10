//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print three_simple_structs
// gdb-check:$1 = struct_in_struct::ThreeSimpleStructs {x: struct_in_struct::Simple {x: 1}, y: struct_in_struct::Simple {x: 2}, z: struct_in_struct::Simple {x: 3}}

// gdb-command:print internal_padding_parent
// gdb-check:$2 = struct_in_struct::InternalPaddingParent {x: struct_in_struct::InternalPadding {x: 4, y: 5}, y: struct_in_struct::InternalPadding {x: 6, y: 7}, z: struct_in_struct::InternalPadding {x: 8, y: 9}}

// gdb-command:print padding_at_end_parent
// gdb-check:$3 = struct_in_struct::PaddingAtEndParent {x: struct_in_struct::PaddingAtEnd {x: 10, y: 11}, y: struct_in_struct::PaddingAtEnd {x: 12, y: 13}, z: struct_in_struct::PaddingAtEnd {x: 14, y: 15}}


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v three_simple_structs
// lldb-check:[...] { x = { x = 1 } y = { x = 2 } z = { x = 3 } }

// lldb-command:v internal_padding_parent
// lldb-check:[...] { x = { x = 4 y = 5 } y = { x = 6 y = 7 } z = { x = 8 y = 9 } }

// lldb-command:v padding_at_end_parent
// lldb-check:[...] { x = { x = 10 y = 11 } y = { x = 12 y = 13 } z = { x = 14 y = 15 } }

// lldb-command:v mixed
// lldb-check:[...] { x = { x = 16 y = 17 } y = { x = 18 y = 19 } z = { x = 20 } w = 21 }

// lldb-command:v bag
// lldb-check:[...] { x = { x = 22 } }

// lldb-command:v bag_in_bag
// lldb-check:[...] { x = { x = { x = 23 } } }

// lldb-command:v tjo
// lldb-check:[...] { x = { x = { x = { x = 24 } } } }

// lldb-command:v tree
// lldb-check:[...] { x = { x = 25 } y = { x = { x = 26 y = 27 } y = { x = 28 y = 29 } z = { x = 30 y = 31 } } z = { x = { x = { x = 32 } } } }

#![allow(unused_variables)]

struct Simple {
    x: i32
}

struct InternalPadding {
    x: i32,
    y: i64
}

struct PaddingAtEnd {
    x: i64,
    y: i32
}

struct ThreeSimpleStructs {
    x: Simple,
    y: Simple,
    z: Simple
}

struct InternalPaddingParent {
    x: InternalPadding,
    y: InternalPadding,
    z: InternalPadding
}

struct PaddingAtEndParent {
    x: PaddingAtEnd,
    y: PaddingAtEnd,
    z: PaddingAtEnd
}

struct Mixed {
    x: PaddingAtEnd,
    y: InternalPadding,
    z: Simple,
    w: i16
}

struct Bag {
    x: Simple
}

struct BagInBag {
    x: Bag
}

struct ThatsJustOverkill {
    x: BagInBag
}

struct Tree {
    x: Simple,
    y: InternalPaddingParent,
    z: BagInBag
}

fn main() {

    let three_simple_structs = ThreeSimpleStructs {
        x: Simple { x: 1 },
        y: Simple { x: 2 },
        z: Simple { x: 3 }
    };

    let internal_padding_parent = InternalPaddingParent {
        x: InternalPadding { x: 4, y: 5 },
        y: InternalPadding { x: 6, y: 7 },
        z: InternalPadding { x: 8, y: 9 }
    };

    let padding_at_end_parent = PaddingAtEndParent {
        x: PaddingAtEnd { x: 10, y: 11 },
        y: PaddingAtEnd { x: 12, y: 13 },
        z: PaddingAtEnd { x: 14, y: 15 }
    };

    let mixed = Mixed {
        x: PaddingAtEnd { x: 16, y: 17 },
        y: InternalPadding { x: 18, y: 19 },
        z: Simple { x: 20 },
        w: 21
    };

    let bag = Bag { x: Simple { x: 22 } };
    let bag_in_bag = BagInBag {
        x: Bag {
            x: Simple { x: 23 }
        }
    };

    let tjo = ThatsJustOverkill {
        x: BagInBag {
            x: Bag {
                x: Simple { x: 24 }
            }
        }
    };

    let tree = Tree {
        x: Simple { x: 25 },
        y: InternalPaddingParent {
            x: InternalPadding { x: 26, y: 27 },
            y: InternalPadding { x: 28, y: 29 },
            z: InternalPadding { x: 30, y: 31 }
        },
        z: BagInBag {
            x: Bag {
                x: Simple { x: 32 }
            }
        }
    };

    zzz(); // #break
}

fn zzz() {()}
