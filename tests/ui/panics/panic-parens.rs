// Fail macros without arguments need to be disambiguated in
// certain positions

//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn bigpanic() {
    while (panic!("oops")) {
        if (panic!()) {
            match (panic!()) {
                () => {}
            }
        }
    }
}

fn main() {
    bigpanic();
}
