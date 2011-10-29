// This isn't really xfailed; it's used by the companionmod.rc test
// xfail-test

fn main() {
    assert a::b::g() == "ralph";
    assert a::c::g() == "nelson";
}