
use std;

fn check_sort(v1: &[mutable int], v2: &[mutable int]) {
    let len = std::vec::len::<int>(v1);
    fn lt(a: &int, b: &int) -> bool { ret a < b; }
    fn equal(a: &int, b: &int) -> bool { ret a == b; }
    let f1 = lt;
    let f2 = equal;
    std::sort::quick_sort3::<int>(f1, f2, v1);
    let i = 0u;
    while i < len { log v2.(i); assert (v2.(i) == v1.(i)); i += 1u; }
}

#[test]
fn test() {
    {
        let v1 = ~[mutable 3, 7, 4, 5, 2, 9, 5, 8];
        let v2 = ~[mutable 2, 3, 4, 5, 5, 7, 8, 9];
        check_sort(v1, v2);
    }
    {
        let v1 = ~[mutable 1, 1, 1];
        let v2 = ~[mutable 1, 1, 1];
        check_sort(v1, v2);
    }
    {
        let v1: [mutable int] = ~[mutable ];
        let v2: [mutable int] = ~[mutable ];
        check_sort(v1, v2);
    }
    { let v1 = ~[mutable 9]; let v2 = ~[mutable 9]; check_sort(v1, v2); }
    {
        let v1 = ~[mutable 9, 3, 3, 3, 9];
        let v2 = ~[mutable 3, 3, 3, 9, 9];
        check_sort(v1, v2);
    }
}
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
