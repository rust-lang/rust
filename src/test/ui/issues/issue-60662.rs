// build-pass (FIXME(62277): could be check-pass?)
// compile-flags: -Z unpretty=hir

#![feature(existential_type)]

trait Animal {
}

fn main() {
    pub existential type ServeFut: Animal;
}
