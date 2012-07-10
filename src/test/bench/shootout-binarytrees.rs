use std;
import std::arena;
import methods = std::arena::arena;

enum tree/& { nil, node(&tree, &tree, int), }

fn item_check(t: &tree) -> int {
    alt *t {
      nil { ret 0; }
      node(left, right, item) {
        ret item + item_check(left) - item_check(right);
      }
    }
}

fn bottom_up_tree(arena: &a.arena::arena, item: int, depth: int) -> &a.tree {
    if depth > 0 {
        ret new(*arena) node(bottom_up_tree(arena, 2 * item - 1, depth - 1),
                             bottom_up_tree(arena, 2 * item, depth - 1),
                             item);
    }
    ret new(*arena) nil;
}

fn main(args: ~[str]) {
    let args = if os::getenv("RUST_BENCH").is_some() {
        ~["", "17"]
    } else if args.len() <= 1u {
        ~["", "8"]
    } else {
        args
    };

    let n = int::from_str(args[1]).get();
    let min_depth = 4;
    let mut max_depth;
    if min_depth + 2 > n {
        max_depth = min_depth + 2;
    } else {
        max_depth = n;
    }

    let stretch_arena = arena::arena();
    let stretch_depth = max_depth + 1;
    let stretch_tree = bottom_up_tree(&stretch_arena, 0, stretch_depth);

    io::println(#fmt("stretch tree of depth %d\t check: %d",
                          stretch_depth,
                          item_check(stretch_tree)));

    let long_lived_arena = arena::arena();
    let long_lived_tree = bottom_up_tree(&long_lived_arena, 0, max_depth);
    let mut depth = min_depth;
    while depth <= max_depth {
        let iterations = int::pow(2, (max_depth - depth + min_depth) as uint);
        let mut chk = 0;
        let mut i = 1;
        while i <= iterations {
            let mut temp_tree = bottom_up_tree(&long_lived_arena, i, depth);
            chk += item_check(temp_tree);
            temp_tree = bottom_up_tree(&long_lived_arena, -i, depth);
            chk += item_check(temp_tree);
            i += 1;
        }
        io::println(#fmt("%d\t trees of depth %d\t check: %d",
                         iterations * 2, depth,
                         chk));
        depth += 2;
    }
    io::println(#fmt("long lived trees of depth %d\t check: %d",
                     max_depth,
                          item_check(long_lived_tree)));
}
