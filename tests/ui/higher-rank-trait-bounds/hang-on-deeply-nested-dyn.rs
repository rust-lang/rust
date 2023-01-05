// normalize-stderr-test: "long-type-\d+" -> "long-type-hash"

fn id(
    f: &dyn Fn(u32),
) -> &dyn Fn(
    &dyn Fn(
        &dyn Fn(
            &dyn Fn(&dyn Fn(&dyn Fn(&dyn Fn(&dyn Fn(&dyn Fn(&dyn Fn(&dyn Fn(&dyn Fn(u32))))))))),
        ),
    ),
) {
    f
    //~^ ERROR mismatched types
}

fn main() {}
