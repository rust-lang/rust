class C {
    let x: uint;

    new(x: uint) {
        self.x = x;
    }
}

fn f<T:copy>(_x: T) {
}

#[warn(err_non_implicitly_copyable_typarams)]
fn main() {
    f(C(1u));
}
