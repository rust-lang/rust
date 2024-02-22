// Testing the display of fixed sized arrays in cdb.

// cdb-only
//@ min-cdb-version: 10.0.18317.1001
//@ compile-flags:-g

// === CDB TESTS ==================================================================================

// cdb-command: g

// cdb-command: dx xs,d
// cdb-check:xs,d             [Type: int [5]]
// cdb-check:    [0]              : 1 [Type: int]
// cdb-check:    [1]              : 2 [Type: int]
// cdb-check:    [2]              : 3 [Type: int]
// cdb-check:    [3]              : 4 [Type: int]
// cdb-check:    [4]              : 5 [Type: int]

// cdb-command: dx ys,d
// cdb-check:ys,d             [Type: int [3]]
// cdb-check:    [0]              : 0 [Type: int]
// cdb-check:    [1]              : 0 [Type: int]
// cdb-check:    [2]              : 0 [Type: int]

fn main() {
    // Fixed-size array (type signature is superfluous)
    let xs: [i32; 5] = [1, 2, 3, 4, 5];

    // All elements can be initialized to the same value
    let ys: [i32; 3] = [0; 3];

    // Indexing starts at 0
    println!("first element of the array: {}", xs[0]);
    println!("second element of the array: {}", xs[1]);

    zzz(); // #break
}

fn zzz() { () }
