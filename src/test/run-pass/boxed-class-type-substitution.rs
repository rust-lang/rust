// Regression test that rustc doesn't recurse infinitely substituting
// the boxed type parameter

type Tree<T> = {
    mut parent: option<T>,
};

fn empty<T>() -> Tree<T> { fail }

struct Box {
    let tree: Tree<@Box>;

    new() {
        self.tree = empty();
    }
}

enum layout_data = {
    mut box: option<@Box>
};

fn main() { }