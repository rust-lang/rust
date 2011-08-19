use std;
import std::int;

tag tree { nil; node(@tree, @tree, int); }

fn item_check(t: @tree) -> int {
    alt *t {
      nil. { ret 0; }
      node(left, right, item) {
        ret item + item_check(left) - item_check(right);
      }
    }
}

fn bottom_up_tree(item: int, depth: int) -> @tree {
    if depth > 0 {
        ret @node(bottom_up_tree(2 * item - 1, depth - 1),
                  bottom_up_tree(2 * item, depth - 1), item);
    } else { ret @nil; }
}

fn main() {
    let n = 8;
    let min_depth = 4;
    let max_depth;
    if min_depth + 2 > n {
        max_depth = min_depth + 2;
    } else { max_depth = n; }
    let stretch_depth = max_depth + 1;
    let stretch_tree = bottom_up_tree(0, stretch_depth);
    log #fmt["stretch tree of depth %d\t check: %d", stretch_depth,
             item_check(stretch_tree)];
    let long_lived_tree = bottom_up_tree(0, max_depth);
    let depth = min_depth;
    while depth <= max_depth {
        let iterations = int::pow(2, max_depth - depth + min_depth as uint);
        let chk = 0;
        let i = 1;
        while i <= iterations {
            let temp_tree = bottom_up_tree(i, depth);
            chk += item_check(temp_tree);
            temp_tree = bottom_up_tree(-i, depth);
            chk += item_check(temp_tree);
            i += 1;
        }
        log #fmt["%d\t trees of depth %d\t check: %d", iterations * 2, depth,
                 chk];
        depth += 2;
    }
    log #fmt["long lived trees of depth %d\t check: %d", max_depth,
             item_check(long_lived_tree)];
}
