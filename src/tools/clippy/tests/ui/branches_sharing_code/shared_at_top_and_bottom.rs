#![deny(clippy::branches_sharing_code, clippy::if_same_then_else)]
#![allow(dead_code)]
#![allow(clippy::uninlined_format_args)]
//@no-rustfix
// branches_sharing_code at the top and bottom of the if blocks

struct DataPack {
    id: u32,
    name: String,
    some_data: Vec<u8>,
}

fn overlapping_eq_regions() {
    let x = 9;

    // Overlap with separator
    if x == 7 {
        //~^ branches_sharing_code

        let t = 7;
        let _overlap_start = t * 2;
        let _overlap_end = 2 * t;
        let _u = 9;
    } else {
        let t = 7;
        let _overlap_start = t * 2;
        let _overlap_end = 2 * t;
        println!("Overlap separator");
        let _overlap_start = t * 2;
        let _overlap_end = 2 * t;
        let _u = 9;
    }

    // Overlap with separator
    if x == 99 {
        //~^ branches_sharing_code

        let r = 7;
        let _overlap_start = r;
        let _overlap_middle = r * r;
        let _overlap_end = r * r * r;
        let z = "end";
    } else {
        let r = 7;
        let _overlap_start = r;
        let _overlap_middle = r * r;
        let _overlap_middle = r * r;
        let _overlap_end = r * r * r;
        let z = "end";
    }
}

fn complexer_example() {
    fn gen_id(x: u32, y: u32) -> u32 {
        let x = x & 0x0000_ffff;
        let y = (y & 0xffff_0000) << 16;
        x | y
    }

    fn process_data(data: DataPack) {
        let _ = data;
    }

    let x = 8;
    let y = 9;
    if (x > 7 && y < 13) || (x + y) % 2 == 1 {
        //~^ branches_sharing_code

        let a = 0xcafe;
        let b = 0xffff00ff;
        let e_id = gen_id(a, b);

        println!("From the a `{}` to the b `{}`", a, b);

        let pack = DataPack {
            id: e_id,
            name: "Player 1".to_string(),
            some_data: vec![0x12, 0x34, 0x56, 0x78, 0x90],
        };
        process_data(pack);
    } else {
        let a = 0xcafe;
        let b = 0xffff00ff;
        let e_id = gen_id(a, b);

        println!("The new ID is '{}'", e_id);

        let pack = DataPack {
            id: e_id,
            name: "Player 1".to_string(),
            some_data: vec![0x12, 0x34, 0x56, 0x78, 0x90],
        };
        process_data(pack);
    }
}

/// This should add a note to the lint msg since the moved expression is not `()`
fn added_note_for_expression_use() -> u32 {
    let x = 9;

    let _ = if x == 7 {
        //~^ branches_sharing_code

        let _ = 19;

        let _splitter = 6;

        x << 2
    } else {
        let _ = 19;

        x << 2
    };

    if x == 9 {
        //~^ branches_sharing_code

        let _ = 17;

        let _splitter = 6;

        x * 4
    } else {
        let _ = 17;

        x * 4
    }
}

fn main() {}

mod issue14873 {
    fn foo() -> i32 {
        todo!()
    }

    macro_rules! qux {
        ($a:ident, $b:ident, $condition:expr) => {
            let mut $a: i32 = foo();
            let mut $b: i32 = foo();
            if $condition {
                "."
            } else {
                ""
            };
            $a = foo();
            $b = foo();
        };
    }

    fn share_on_top_and_bottom() {
        if false {
            qux!(a, b, a == b);
        } else {
            qux!(a, b, a != b);
        };

        if false {
            //~^ branches_sharing_code
            let x = 1;
            qux!(a, b, a == b);
            let y = 1;
        } else {
            let x = 1;
            qux!(a, b, a != b);
            let y = 1;
        }
    }
}
