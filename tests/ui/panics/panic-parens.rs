// Fail macros without arguments need to be disambiguated in
// certain positions

//@ run-fail
//@ error-pattern:oops
//@ ignore-emscripten no processes

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
