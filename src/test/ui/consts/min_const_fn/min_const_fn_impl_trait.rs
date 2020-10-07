// gate-test-const_impl_trait

struct AlanTuring<T>(T);
const fn no_rpit2() -> AlanTuring<impl std::fmt::Debug> { //~ `impl Trait`
    AlanTuring(0)
}

const fn no_rpit() -> impl std::fmt::Debug {} //~ `impl Trait`

fn main() {}
