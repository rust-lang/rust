
use std;
import std::int;

tag tree { nil; node(@tree, @tree, int); }

fn item_check(@tree t) -> int {
    alt (*t) {
        case (nil) { ret 0; }
        case (node(?left, ?right, ?item)) {
            ret item + item_check(left) - item_check(right);
        }
    }
}

fn bottom_up_tree(int item, int depth) -> @tree {
    if (depth > 0) {
        ret @node(bottom_up_tree(2 * item - 1, depth - 1),
                  bottom_up_tree(2 * item, depth - 1), item);
    } else { ret @nil; }
}

fn main() {
    auto n = 8;
    auto min_depth = 4;
    auto max_depth;
    if (min_depth + 2 > n) {
        max_depth = min_depth + 2;
    } else { max_depth = n; }
    auto stretch_depth = max_depth + 1;
    auto stretch_tree = bottom_up_tree(0, stretch_depth);
    log #fmt("stretch tree of depth %d\t check: %d", stretch_depth,
             item_check(stretch_tree));
    auto long_lived_tree = bottom_up_tree(0, max_depth);
    auto depth = min_depth;
    while (depth <= max_depth) {
        auto iterations = int::pow(2, max_depth - depth + min_depth as uint);
        auto chk = 0;
        auto i = 1;
        while (i <= iterations) {
            auto temp_tree = bottom_up_tree(i, depth);
            chk += item_check(temp_tree);
            temp_tree = bottom_up_tree(-i, depth);
            chk += item_check(temp_tree);
            i += 1;
        }
        log #fmt("%d\t trees of depth %d\t check: %d", iterations * 2, depth,
                 chk);
        depth += 2;
    }
    log #fmt("long lived trees of depth %d\t check: %d", max_depth,
             item_check(long_lived_tree));
}