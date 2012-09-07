// Regression test that rustc doesn't recurse infinitely substituting
// the boxed type parameter

type Tree<T> = {
    mut parent: Option<T>,
};

fn empty<T>() -> Tree<T> { fail }

struct Box {
    tree: Tree<@Box>
}

fn Box() -> Box {
    Box {
        tree: empty()
    }
}

enum layout_data = {
    mut box: Option<@Box>
};

fn main() { }