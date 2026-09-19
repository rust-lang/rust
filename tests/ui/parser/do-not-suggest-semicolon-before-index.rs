fn example() {
    (foo[
        bar
        .baz(), //~ ERROR: expected one of `.`, `?`, `]`, or an operator, found `,`
    ])
}

fn main() {}
