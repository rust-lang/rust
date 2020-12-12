// run-pass

// Test disjoint capture within an impl block

#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| NOTE: `#[warn(incomplete_features)]` on by default
//~| NOTE: see issue #53488 <https://github.com/rust-lang/rust/issues/53488>

struct Filter {
    div: i32,
}
impl Filter {
    fn allowed(&self, x: i32) -> bool {
        x % self.div == 1
    }
}

struct Data {
    filter: Filter,
    list: Vec<i32>,
}
impl Data {
    fn update(&mut self) {
        // The closure passed to filter only captures self.filter,
        // therefore mutating self.list is allowed.
        self.list.retain(
            |v| self.filter.allowed(*v),
        );
    }
}

fn main() {
    let mut d = Data { filter: Filter { div: 3 }, list: Vec::new() };

    for i in 1..10 {
        d.list.push(i);
    }

    d.update();
}
