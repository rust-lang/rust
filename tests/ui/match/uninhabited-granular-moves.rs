// See rust-lang/rust#146590, as well as Zulip discussion:
//
// https://rust-lang.zulipchat.com/#narrow/channel/513289-t-patterns/topic/Question.20about.20patterns.20and.20moves/with/558638455
//
// Whether pattern matching performs a discriminant read shouldn't depend on whether
// you explicitly write down an uninhabited branch, or leave it implicit.

enum Emp { }

enum Foo<A> {
    Bar(A),
    Qux(Emp),
}

fn test1(thefoo: Foo<(Box<u64>, Box<u64>)>) {
    match thefoo {
        Foo::Bar((a, _)) => { }
    }

    match thefoo {
        Foo::Bar((_, a)) => { }
    }
}

fn test2(thefoo: Foo<(Box<u64>, Box<u64>)>) {
    match thefoo {
        Foo::Bar((a, _)) => { }
        Foo::Qux(_) => { }
    }
    match thefoo { //~ ERROR: use of partially moved value: `thefoo`
        Foo::Bar((_, a)) => { }
        Foo::Qux(_) => { }
    }
}

fn test3(thefoo: Foo<(Box<u64>, Box<u64>)>) {
    match thefoo {
        Foo::Bar((a, _)) => { }
        Foo::Qux(_) => { }
    }
    match thefoo {
        Foo::Bar((_, a)) => { }
    }
}

fn main() {}
