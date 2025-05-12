// This file checks that fn ptrs are *not* considered structurally matchable.
// See also rust-lang/rust#63479 and RFC 3535.

fn main() {
    let mut count = 0;

    // A type which is not structurally matchable:
    struct NotSM;

    // And one that is:
    #[derive(PartialEq, Eq)]
    struct SM;

    fn trivial() {}

    fn sm_to(_: SM) {}
    fn not_sm_to(_: NotSM) {}
    fn to_sm() -> SM { SM }
    fn to_not_sm() -> NotSM { NotSM }

    // To recreate the scenario of interest in #63479, we need to add
    // a ref-level-of-indirection so that we descend into the type.

    fn r_sm_to(_: &SM) {}
    fn r_not_sm_to(_: &NotSM) {}
    fn r_to_r_sm(_: &()) -> &SM { &SM }
    fn r_to_r_not_sm(_: &()) -> &NotSM { &NotSM }

    #[derive(PartialEq, Eq)]
    struct Wrap<T>(T);

    // In the code below, we put the match input into a local so that
    // we can assign it an explicit type that is an fn ptr instead of
    // a singleton type of the fn itself that the type inference would
    // otherwise assign.

    // Check that fn() is structural-match
    const CFN1: Wrap<fn()> = Wrap(trivial);
    let input: Wrap<fn()> = Wrap(trivial);
    match Wrap(input) {
        Wrap(CFN1) => count += 1, //~ERROR behave unpredictably
        Wrap(_) => {}
    };

    // Check that fn(T) is structural-match when T is too.
    const CFN2: Wrap<fn(SM)> = Wrap(sm_to);
    let input: Wrap<fn(SM)> = Wrap(sm_to);
    match Wrap(input) {
        Wrap(CFN2) => count += 1, //~ERROR behave unpredictably
        Wrap(_) => {}
    };

    // Check that fn() -> T is structural-match when T is too.
    const CFN3: Wrap<fn() -> SM> = Wrap(to_sm);
    let input: Wrap<fn() -> SM> = Wrap(to_sm);
    match Wrap(input) {
        Wrap(CFN3) => count += 1, //~ERROR behave unpredictably
        Wrap(_) => {}
    };

    // Check that fn(T) is structural-match even if T is not.
    const CFN4: Wrap<fn(NotSM)> = Wrap(not_sm_to);
    let input: Wrap<fn(NotSM)> = Wrap(not_sm_to);
    match Wrap(input) {
        Wrap(CFN4) => count += 1, //~ERROR behave unpredictably
        Wrap(_) => {}
    };

    // Check that fn() -> T is structural-match even if T is not.
    const CFN5: Wrap<fn() -> NotSM> = Wrap(to_not_sm);
    let input: Wrap<fn() -> NotSM> = Wrap(to_not_sm);
    match Wrap(input) {
        Wrap(CFN5) => count += 1, //~ERROR behave unpredictably
        Wrap(_) => {}
    };

    // Check that fn(&T) is structural-match when T is too.
    const CFN6: Wrap<fn(&SM)> = Wrap(r_sm_to);
    let input: Wrap<fn(&SM)> = Wrap(r_sm_to);
    match Wrap(input) {
        Wrap(CFN6) => count += 1, //~ERROR behave unpredictably
        Wrap(_) => {}
    };

    // Check that fn() -> &T is structural-match when T is too.
    const CFN7: Wrap<fn(&()) -> &SM> = Wrap(r_to_r_sm);
    let input: Wrap<fn(&()) -> &SM> = Wrap(r_to_r_sm);
    match Wrap(input) {
        Wrap(CFN7) => count += 1, //~ERROR behave unpredictably
        Wrap(_) => {}
    };

    // Check that fn(T) is structural-match even if T is not.
    const CFN8: Wrap<fn(&NotSM)> = Wrap(r_not_sm_to);
    let input: Wrap<fn(&NotSM)> = Wrap(r_not_sm_to);
    match Wrap(input) {
        Wrap(CFN8) => count += 1, //~ERROR behave unpredictably
        Wrap(_) => {}
    };

    // Check that fn() -> T is structural-match even if T is not.
    const CFN9: Wrap<fn(&()) -> &NotSM> = Wrap(r_to_r_not_sm);
    let input: Wrap<fn(&()) -> &NotSM> = Wrap(r_to_r_not_sm);
    match Wrap(input) {
        Wrap(CFN9) => count += 1, //~ERROR behave unpredictably
        Wrap(_) => {}
    };

    // Check that a type which has fn ptrs is structural-match.
    #[derive(PartialEq, Eq)]
    struct Foo {
        alpha: fn(NotSM),
        beta: fn() -> NotSM,
        gamma: fn(SM),
        delta: fn() -> SM,
    }

    const CFOO: Foo = Foo {
        alpha: not_sm_to,
        beta: to_not_sm,
        gamma: sm_to,
        delta: to_sm,
    };

    let input = Foo { alpha: not_sm_to, beta: to_not_sm, gamma: sm_to, delta: to_sm };
    match input {
        CFOO => count += 1, //~ERROR behave unpredictably
        Foo { .. } => {}
    };

    // Final count must be 10 now if all
    assert_eq!(count, 10);
}
