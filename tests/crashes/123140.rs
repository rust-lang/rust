//@ known-bug: #123140
trait Project {
    const SELF: Self;
}

fn take1(_: Project<SELF = { loop {} }>) {}
