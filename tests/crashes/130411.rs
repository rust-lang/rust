//@ known-bug: #130411
trait Project {
    const SELF: Self;
}

fn take1(_: Project<SELF = {}>) {}
