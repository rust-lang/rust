fn main() {

    // The Internet made me do it.

    obj cat() {
        fn ack() -> str {
            ret "ack";
        }
        fn meow() -> str {
            ret "meow";
        }
        fn zzz() -> str {
            ret self.meow();
        }
    }

    let shortcat = cat();

    let longcat = obj() {
        fn lol() -> str {
            ret "lol";
        }
        fn nyan() -> str {
            ret "nyan";
        }
        with shortcat
    };

    let longercat = obj() {
        fn meow() -> str {
            ret "zzz";
        }
        with shortcat
    };

    let evenlongercat = obj() {
        fn meow() -> str {
            ret "zzzzzz";
        }
        with longercat
    };

    // Tests self-call.
    assert (shortcat.zzz() == "meow");

    // Tests forwarding/backwarding + self-call.
    assert (longcat.zzz() == "meow");

    // Tests forwarding/backwarding + self-call + override.
    assert (longercat.zzz() == "zzz");

    // Tests two-level forwarding/backwarding + self-call + override.
    assert (evenlongercat.zzz() == "zzzzzz");
}
