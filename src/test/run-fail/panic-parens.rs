// Fail macros without arguments need to be disambiguated in
// certain positions
// error-pattern:oops

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
