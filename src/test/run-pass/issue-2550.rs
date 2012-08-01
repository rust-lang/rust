class C {
    let x: uint;

    new(x: uint) {
        self.x = x;
    }
}

fn f<T:copy>(_x: T) {
}

#[deny(non_implicitly_copyable_typarams)]
fn main() {
    f(C(1u));
}
