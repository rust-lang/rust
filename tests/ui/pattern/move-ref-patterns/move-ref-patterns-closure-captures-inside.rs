fn main() {
    struct S; // Not `Copy`.

    let mut tup0 = (S, S);
    let mut tup1 = (S, S, S);
    let tup2 = (S, S);
    let tup3 = (S, S, S);
    let tup4 = (S, S);
    let mut arr0 = [S, S, S];
    let mut arr1 = [S, S, S, S, S];
    let arr2 = [S, S, S];
    let arr3 = [S, S, S, S, S];

    // The `mov` bindings require that we capture the scrutinees by-move.
    let mut closure = || {
        // Tuples...
        let (ref mut borrow, mov) = tup0;
        let (mov, _, ref mut borrow) = tup1;
        let (ref borrow, mov) = tup2;
        let (mov, _, ref borrow) = tup3;
        let (ref borrow, mov) = tup4;
        // Arrays...
        let [mov @ .., ref borrow] = arr0;
        let [_, ref mut borrow @ .., _, mov] = arr1;
        let [mov @ .., ref borrow] = arr2;
        let [_, ref borrow @ .., _, mov] = arr3;
    };

    // Now we try to borrow and move the captures, which should result in errors...
    // ...for tuples:
    drop(&tup0); //~ ERROR borrow of moved value: `tup0`
    drop(&tup1); //~ ERROR borrow of moved value: `tup1`
    drop(&tup2); //~ ERROR borrow of moved value: `tup2`
    drop(&tup3); //~ ERROR borrow of moved value: `tup3`
    // Ostensibly this should compile.
    // However, because closures don't capture individual fields, which is changed in RFC 2229,
    // this won't compile because the entire product is moved into the closure.
    // The same applies to the array patterns below.
    drop(&tup4.0); //~ ERROR borrow of moved value: `tup4`
    // ...for arrays:
    drop(&arr0); //~ ERROR borrow of moved value: `arr0`
    let [_, mov1, mov2, mov3, _] = &arr1; //~ ERROR borrow of moved value: `arr1`
    drop(&arr2); //~ ERROR borrow of moved value: `arr2`
    let [_, mov1, mov2, mov3, _] = &arr3; //~ ERROR borrow of moved value: `arr3`

    // Let's redo ^--- with a `match` + sum type:
    macro_rules! m {
        ($p:pat = $s:expr) => {
            match $s {
                Some($p) => {}
                _ => {}
            }
        };
    }
    let mut tup0: Option<(S, S)> = None;
    let mut tup1: Option<(S, S, S)> = None;
    let tup2: Option<(S, S)> = None;
    let tup3: Option<(S, S, S)> = None;
    let tup4: Option<(S, S)> = None;
    let mut arr0: Option<[S; 3]> = None;
    let mut arr1: Option<[S; 5]> = None;
    let arr2: Option<[S; 3]> = None;
    let arr3: Option<[S; 5]> = None;
    let mut closure = || {
        m!((ref mut borrow, mov) = tup0);
        m!((mov, _, ref mut borrow) = tup1);
        m!((ref borrow, mov) = tup2);
        m!((mov, _, ref borrow) = tup3);
        m!((ref borrow, mov) = tup4);
        m!([mov @ .., ref borrow] = arr0);
        m!([_, ref mut borrow @ .., _, mov] = arr1);
        m!([mov @ .., ref borrow] = arr2);
        m!([_, ref borrow @ .., _, mov] = arr3);
    };
    drop(&tup0); //~ ERROR borrow of moved value: `tup0`
    drop(&tup1); //~ ERROR borrow of moved value: `tup1`
    drop(&tup2); //~ ERROR borrow of moved value: `tup2`
    drop(&tup3); //~ ERROR borrow of moved value: `tup3`
    m!((ref x, _) = &tup4); //~ ERROR borrow of moved value: `tup4`
    drop(&arr0); //~ ERROR borrow of moved value: `arr0`
    m!([_, mov1, mov2, mov3, _] = &arr1); //~ ERROR borrow of moved value: `arr1`
    drop(&arr2); //~ ERROR borrow of moved value: `arr2`
    m!([_, mov1, mov2, mov3, _] = &arr3); //~ ERROR borrow of moved value: `arr3`

    // Let's redo ^--- with `if let` (which may diverge from `match` in the future):
    macro_rules! m {
        ($p:pat = $s:expr) => {
            if let Some($p) = $s {}
        };
    }
    let mut tup0: Option<(S, S)> = None;
    let mut tup1: Option<(S, S, S)> = None;
    let tup2: Option<(S, S)> = None;
    let tup3: Option<(S, S, S)> = None;
    let tup4: Option<(S, S)> = None;
    let mut arr0: Option<[S; 3]> = None;
    let mut arr1: Option<[S; 5]> = None;
    let arr2: Option<[S; 3]> = None;
    let arr3: Option<[S; 5]> = None;
    let mut closure = || {
        m!((ref mut borrow, mov) = tup0);
        m!((mov, _, ref mut borrow) = tup1);
        m!((ref borrow, mov) = tup2);
        m!((mov, _, ref borrow) = tup3);
        m!((ref borrow, mov) = tup4);
        m!([mov @ .., ref borrow] = arr0);
        m!([_, ref mut borrow @ .., _, mov] = arr1);
        m!([mov @ .., ref borrow] = arr2);
        m!([_, ref borrow @ .., _, mov] = arr3);
    };
    drop(&tup0); //~ ERROR borrow of moved value: `tup0`
    drop(&tup1); //~ ERROR borrow of moved value: `tup1`
    drop(&tup2); //~ ERROR borrow of moved value: `tup2`
    drop(&tup3); //~ ERROR borrow of moved value: `tup3`
    m!((ref x, _) = &tup4); //~ ERROR borrow of moved value: `tup4`
    drop(&arr0); //~ ERROR borrow of moved value: `arr0`
    m!([_, mov1, mov2, mov3, _] = &arr1); //~ ERROR borrow of moved value: `arr1`
    drop(&arr2); //~ ERROR borrow of moved value: `arr2`
    m!([_, mov1, mov2, mov3, _] = &arr3); //~ ERROR borrow of moved value: `arr3`
}
