// -*- rust -*-

// error-pattern: attempted dynamic environment-capture

fn main() {

    fn foo() -> int {
        ret bar;
    }

    let int bar = 5;
}
