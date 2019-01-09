// ignore-tidy-linelength
// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print three_simple_structs
// gdbg-check:$1 = {x = {x = 1}, y = {x = 2}, z = {x = 3}}
// gdbr-check:$1 = struct_in_struct::ThreeSimpleStructs {x: struct_in_struct::Simple {x: 1}, y: struct_in_struct::Simple {x: 2}, z: struct_in_struct::Simple {x: 3}}

// gdb-command:print internal_padding_parent
// gdbg-check:$2 = {x = {x = 4, y = 5}, y = {x = 6, y = 7}, z = {x = 8, y = 9}}
// gdbr-check:$2 = struct_in_struct::InternalPaddingParent {x: struct_in_struct::InternalPadding {x: 4, y: 5}, y: struct_in_struct::InternalPadding {x: 6, y: 7}, z: struct_in_struct::InternalPadding {x: 8, y: 9}}

// gdb-command:print padding_at_end_parent
// gdbg-check:$3 = {x = {x = 10, y = 11}, y = {x = 12, y = 13}, z = {x = 14, y = 15}}
// gdbr-check:$3 = struct_in_struct::PaddingAtEndParent {x: struct_in_struct::PaddingAtEnd {x: 10, y: 11}, y: struct_in_struct::PaddingAtEnd {x: 12, y: 13}, z: struct_in_struct::PaddingAtEnd {x: 14, y: 15}}


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print three_simple_structs
// lldbg-check:[...]$0 = ThreeSimpleStructs { x: Simple { x: 1 }, y: Simple { x: 2 }, z: Simple { x: 3 } }
// lldbr-check:(struct_in_struct::ThreeSimpleStructs) three_simple_structs = ThreeSimpleStructs { x: Simple { x: 1 }, y: Simple { x: 2 }, z: Simple { x: 3 } }

// lldb-command:print internal_padding_parent
// lldbg-check:[...]$1 = InternalPaddingParent { x: InternalPadding { x: 4, y: 5 }, y: InternalPadding { x: 6, y: 7 }, z: InternalPadding { x: 8, y: 9 } }
// lldbr-check:(struct_in_struct::InternalPaddingParent) internal_padding_parent = InternalPaddingParent { x: InternalPadding { x: 4, y: 5 }, y: InternalPadding { x: 6, y: 7 }, z: InternalPadding { x: 8, y: 9 } }

// lldb-command:print padding_at_end_parent
// lldbg-check:[...]$2 = PaddingAtEndParent { x: PaddingAtEnd { x: 10, y: 11 }, y: PaddingAtEnd { x: 12, y: 13 }, z: PaddingAtEnd { x: 14, y: 15 } }
// lldbr-check:(struct_in_struct::PaddingAtEndParent) padding_at_end_parent = PaddingAtEndParent { x: PaddingAtEnd { x: 10, y: 11 }, y: PaddingAtEnd { x: 12, y: 13 }, z: PaddingAtEnd { x: 14, y: 15 } }

// lldb-command:print mixed
// lldbg-check:[...]$3 = Mixed { x: PaddingAtEnd { x: 16, y: 17 }, y: InternalPadding { x: 18, y: 19 }, z: Simple { x: 20 }, w: 21 }
// lldbr-check:(struct_in_struct::Mixed) mixed = Mixed { x: PaddingAtEnd { x: 16, y: 17 }, y: InternalPadding { x: 18, y: 19 }, z: Simple { x: 20 }, w: 21 }

// lldb-command:print bag
// lldbg-check:[...]$4 = Bag { x: Simple { x: 22 } }
// lldbr-check:(struct_in_struct::Bag) bag = Bag { x: Simple { x: 22 } }

// lldb-command:print bag_in_bag
// lldbg-check:[...]$5 = BagInBag { x: Bag { x: Simple { x: 23 } } }
// lldbr-check:(struct_in_struct::BagInBag) bag_in_bag = BagInBag { x: Bag { x: Simple { x: 23 } } }

// lldb-command:print tjo
// lldbg-check:[...]$6 = ThatsJustOverkill { x: BagInBag { x: Bag { x: Simple { x: 24 } } } }
// lldbr-check:(struct_in_struct::ThatsJustOverkill) tjo = ThatsJustOverkill { x: BagInBag { x: Bag { x: Simple { x: 24 } } } }

// lldb-command:print tree
// lldbg-check:[...]$7 = Tree { x: Simple { x: 25 }, y: InternalPaddingParent { x: InternalPadding { x: 26, y: 27 }, y: InternalPadding { x: 28, y: 29 }, z: InternalPadding { x: 30, y: 31 } }, z: BagInBag { x: Bag { x: Simple { x: 32 } } } }
// lldbr-check:(struct_in_struct::Tree) tree = Tree { x: Simple { x: 25 }, y: InternalPaddingParent { x: InternalPadding { x: 26, y: 27 }, y: InternalPadding { x: 28, y: 29 }, z: InternalPadding { x: 30, y: 31 } }, z: BagInBag { x: Bag { x: Simple { x: 32 } } } }

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

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
