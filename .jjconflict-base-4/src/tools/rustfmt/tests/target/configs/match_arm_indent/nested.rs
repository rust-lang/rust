// rustfmt-match_arm_indent: false

fn r#match() {
    match value {
    Arm::Prev => f(),
    // inline match
    ModeratelyLongOption(n) => match n {
    A => f(),
    B => {
        1;
        2;
        3;
    }
    AnotherLongerOption => {
        1;
        2;
    }
    _ if there || is || a || guard => {
        nothing_changes();
    }
    },
    Arm::Next => {
        1;
        2;
        3;
    }
    }
}

// things which break up the nested match arm
fn r#break() {
    match value {
    Arm::Prev => f(),
    // inline match
    ModeratelyLongOption(n) =>
    {
        #[attr]
        match n {
        A => f(),
        B => c(),
        C => 1,
        }
    }
    Arm::Next => n(),
    Two | Patterns =>
    /* inline comment */
    {
        match val {
        C => 3,
        D => func(),
        }
    }
    Arm::Last => l(),
    }
}

fn parens() {
    let result = Some(Other(match value {
    Option1 => 1,
    Option2 => {
        stuff();
        2
    }
    }));
}

fn silly() {
    match value {
    Inner(i1) => match i1 {
    Inner(i2) => match i2 {
    Inner(i3) => match i3 {
    Inner(i4) => match i4 {
    Inner => "it's a readability tradeoff, really",
    },
    },
    },
    },
    }
}
