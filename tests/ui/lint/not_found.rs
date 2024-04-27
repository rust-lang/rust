//@ check-pass

// this tests the `unknown_lint` lint, especially the suggestions

// the suggestion only appears if a lint with the lowercase name exists
#[allow(FOO_BAR)]
//~^ WARNING unknown lint

// the suggestion appears on all-uppercase names
#[warn(DEAD_CODE)]
//~^ WARNING unknown lint
//~| HELP did you mean

// the suggestion appears also on mixed-case names
#[deny(Warnings)]
//~^ WARNING unknown lint
//~| HELP did you mean

fn main() {
    unimplemented!();
}
