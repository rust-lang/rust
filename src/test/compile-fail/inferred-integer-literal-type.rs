// Issue #1425.

// Relax the need for the "u" suffix on unsigned integer literals
// under certain very limited conditions.

// But, when inferring uint types for would-be ints, make sure we
// don't allow chars to be treated as uints.

fn main() {
    let x1: uint = 'c'; //! ERROR mismatched types
    let x2 = 1u + 'c'; //! ERROR mismatched types
    let x3: uint = 1u + 'c'; //! ERROR mismatched types
    let x4 = vec::slice(["hello", "world"], 'c', 'd');
    //!^ ERROR mismatched types
    //!^^ ERROR mismatched types
    let x5 = vec::slice(["hello", "world"], 0u, 'c');
    //!^ ERROR mismatched types
    let x6 = vec::slice(["hello", "world"], 'c', 1u);
    //!^ ERROR mismatched types
}
