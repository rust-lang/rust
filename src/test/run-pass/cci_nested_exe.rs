// xfail-fast - check-fast doesn't understand aux-build
// aux-build:cci_nested_lib.rs

use cci_nested_lib;
import cci_nested_lib::*;

fn main() {
    let lst = new_int_alist();
    alist_add(lst, 22, "hi");
    alist_add(lst, 44, "ho");
    assert alist_get(lst, 22) == "hi";
    assert alist_get(lst, 44) == "ho";

    let lst = new_int_alist_2();
    alist_add(lst, 22, "hi");
    alist_add(lst, 44, "ho");
    assert alist_get(lst, 22) == "hi";
    assert alist_get(lst, 44) == "ho";
}
