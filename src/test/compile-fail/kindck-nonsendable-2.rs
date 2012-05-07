fn foo(_x: r) {}

resource r(_x: ()) {}

fn main() {
    let x = r(());
    let _ = fn~() {
        // Error even though this is the last use:
        foo(x); //! ERROR not a sendable value
    };

    let x = r(());
    let _ = fn@() {
        // OK in fn@ because this is the last use:
        foo(x);
    };
}