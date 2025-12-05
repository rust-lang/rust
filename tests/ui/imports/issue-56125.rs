//@ edition:2018
//@ compile-flags:--extern issue_56125
//@ aux-build:issue-56125.rs

mod m1 {
    use issue_56125::last_segment::*;
    //~^ ERROR `issue_56125` is ambiguous
}

mod m2 {
    use issue_56125::non_last_segment::non_last_segment::*;
    //~^ ERROR `issue_56125` is ambiguous
}

mod m3 {
    mod empty {}
    use empty::issue_56125; //~ ERROR unresolved import `empty::issue_56125`
    use issue_56125::*; //~ ERROR `issue_56125` is ambiguous
}

fn main() {}
