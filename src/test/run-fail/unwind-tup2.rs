// error-pattern:fail

fn fold_local() -> @[int]{
    @[0,0,0,0,0,0]
}

fn fold_remote() -> @[int]{
    fail;
}

fn main() {
    let lss = (fold_local(), fold_remote());
}
