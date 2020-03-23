struct AlanTuring<T>(T);
const fn no_rpit2() -> AlanTuring<impl std::fmt::Debug> {
    //~^ ERROR `impl Trait` in const fn is unstable
    AlanTuring(0)
}

const fn no_rpit() -> impl std::fmt::Debug {} //~ ERROR `impl Trait` in const fn is unstable

fn main() {}
