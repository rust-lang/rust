// compile-pass
// compile-flags: -Z unpretty=hir

#![feature(existential_type)]

trait Animal {
}

fn main() {
    pub existential type ServeFut: Animal;
}
