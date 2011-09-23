// error-pattern:fail

fn failfn() {
    fail;
}

fn main() {
    ~0;
    failfn();
}