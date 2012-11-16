// Fail statements without arguments need to be disambiguated in
// certain positions
// error-pattern:oops

fn bigfail() {
    while (fail ~"oops") { if (fail) {
        match (fail) { () => {
        }
                     }
    }};
}

fn main() { bigfail(); }
