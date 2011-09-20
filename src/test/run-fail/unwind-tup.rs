// error-pattern:fail

fn fold_local() -> @[int]{
    fail;
}

fn main() {
    let lss = (fold_local(), 0);
}
