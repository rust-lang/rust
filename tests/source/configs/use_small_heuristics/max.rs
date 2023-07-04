// rustfmt-use_small_heuristics: Max

enum Lorem {
    Ipsum,
    Dolor(bool),
    Sit {
        amet: Consectetur,
        adipiscing: Elit,
    },
}

fn main() {
    lorem("lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing");

    let lorem = Lorem {
        ipsum: dolor,
        sit: amet,
    };

    let lorem = if ipsum {
        dolor
    } else {
        sit
    };
}

fn format_let_else() {
    let Some(a) = opt else {};

    let Some(b) = opt else { return };

    let Some(c) = opt else { return };

    let Some(d) = some_very_very_very_very_long_name else { return };
}
